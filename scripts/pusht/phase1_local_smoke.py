"""Phase 1 local-only Push-T C-JEPA smoke test.

This script is the first MVP step in the FRD:
- load a Push-T slot sample locally
- run C-JEPA in Python
- inspect the output shape

It intentionally avoids any client/server layer.

Prerequisites:
- run it from the repo-local .venv
- have the local Python deps available in that venv: torch, numpy, einops,
    huggingface_hub, timm, and the repo's checkpoint-compatible modules
- allow the script to download the default Push-T checkpoint on first run, or
    pass a local compatible checkpoint path
- have network access for the first checkpoint download if you are not
    pointing at an existing file
- optionally provide a Push-T slot pickle if you want to test a real sample
    instead of --use-random-input
- make sure the repo has write access to checkpoints/ and outputs/ so it can
    cache the model and write the JSON summary

Relevant keywords:
- checkpoints: Cache location for serialized weights. The smoke test reuses
    the same artifact on later runs, so you only pay the download cost once and
    can inspect a stable local file.
- Slots: The latent object tokens consumed by the predictor. Each slot
    corresponds to one tracked entity across time, which is the unit this
    Push-T model actually reasons over.
- .venv: The repo-local virtual environment. It pins the exact Python packages
    needed to deserialize the checkpoint and run inference without contaminating
    the system interpreter.
- torch: The PyTorch runtime used for tensor ops, checkpoint deserialization,
    and local inference. This is the core execution engine behind the smoke test.
- numpy: The array library used for tensor-adjacent data handling and
    serialization compatibility in the repo. It fills in the usual glue code
    around PyTorch.
- einops: The tensor reshaping utility used by the model code to keep
    dimension transforms explicit and correct. It prevents shape juggling from
    turning into unreadable indexing logic.
- huggingface_hub: The client used to download the default checkpoint from
    Hugging Face on demand. It lets the script fetch the model automatically
    instead of requiring manual setup.
- timm: A transitive dependency expected by the checkpoint's imported modules
    during unpickling. The saved object references it, so the environment must
    have it available.
- custom_models: A compatibility import path for legacy checkpoint objects
    saved before the repo was reorganized. The shim keeps old serialized
    references from failing during load.
- videosaur: Another legacy import path required so old saved objects can
    resolve their module references. Without it, unpickling the checkpoint would
    stop before inference starts.
- stable_worldmodel: Compatibility package that exposes the world-model classes
    under historical import names. It bridges the checkpoint's saved module paths
    to the current code layout.
- Push-T: The downstream benchmark and data domain used to validate the
    slot-based predictor on tracked-object dynamics. It is the practical target
    this smoke test is meant to support.
- C-JEPA: The compact world-model predictor exercised by this smoke test to
    verify loading and forward inference. If this works locally, the checkpoint
    wiring is correct.
- slot pickle: A serialized bank of pre-extracted slot tensors used for
    real-data smoke testing instead of synthetic input. It lets you test the
    model on actual latent inputs.
- outputs: The directory where the run summary JSON is written after
    inference completes. It gives you a small artifact you can inspect or feed
    into automation.

Original sources:
- VideoSAUR paper: https://arxiv.org/abs/2306.04829
- VideoSAUR repository: https://github.com/martius-lab/videosaur
- VideoSAUR project page: https://martius-lab.github.io/videosaur
- Push-T C-JEPA checkpoint source: https://huggingface.co/HazelNam/CJEPA
- Stable-Pretraining: https://galilai-group.github.io/stable-pretraining/
- Stable-WorldModel: https://galilai-group.github.io/stable-worldmodel/

to run: d:\Soqqle\v2\soqqcjepa\.venv\Scripts\python.exe phase1_local_smoke.py --use-random-input
"""

from __future__ import annotations

import argparse
import json
import pickle as pkl
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cjepa_predictor import MaskedSlotPredictor

DEFAULT_NUM_SLOTS = 4
DEFAULT_SLOT_DIM = 128
DEFAULT_HISTORY_FRAMES = 5
DEFAULT_PRED_FRAMES = 3
DEFAULT_NUM_MASKED_SLOTS = 2
DEFAULT_DEPTH = 6
DEFAULT_HEADS = 16
DEFAULT_DIM_HEAD = 64
DEFAULT_MLP_DIM = 2048
DEFAULT_DROPOUT = 0.1
DEFAULT_CHECKPOINT_URL = "https://huggingface.co/HazelNam/CJEPA/resolve/main/cjepa-ckpts/pusht_videosaur_1_epoch_30_object.ckpt"
DEFAULT_CHECKPOINT_NAME = "pusht_videosaur_1_epoch_30_object.ckpt"


def build_predictor(args: argparse.Namespace) -> MaskedSlotPredictor:
    """Build the local C-JEPA predictor used for the smoke test."""
    predictor = MaskedSlotPredictor(
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        history_frames=args.history_frames,
        pred_frames=args.pred_frames,
        num_masked_slots=args.num_masked_slots,
        seed=args.seed,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    )
    return predictor


def infer_checkpoint_spec(checkpoint_obj) -> dict[str, int]:
    """Read the model dimensions directly from the loaded checkpoint object."""
    if hasattr(checkpoint_obj, "model") and hasattr(checkpoint_obj.model, "predictor"):
        predictor_module = checkpoint_obj.model.predictor
    elif hasattr(checkpoint_obj, "predictor"):
        predictor_module = checkpoint_obj.predictor
    else:
        predictor_module = checkpoint_obj

    time_pos_embed = getattr(predictor_module, "time_pos_embed", None)
    if time_pos_embed is None:
        raise AttributeError("Checkpoint does not expose a predictor with time_pos_embed")

    total_frames = int(time_pos_embed.shape[1])
    history_frames = int(getattr(predictor_module, "history_frames", total_frames - 1))
    pred_frames = int(getattr(predictor_module, "pred_frames", total_frames - history_frames))

    return {
        "num_slots": int(getattr(predictor_module, "num_slots")),
        "slot_dim": int(getattr(predictor_module, "slot_dim")),
        "history_frames": history_frames,
        "pred_frames": pred_frames,
    }


def download_checkpoint(destination: Path, url: str = DEFAULT_CHECKPOINT_URL) -> Path:
    """Download the default Push-T checkpoint into the local cache directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = destination.parent / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase1] Downloading checkpoint from Hugging Face: {url}")
    downloaded_path = Path(
        hf_hub_download(
            repo_id="HazelNam/CJEPA",
            filename="cjepa-ckpts/pusht_videosaur_1_epoch_30_object.ckpt",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            force_download=True,
        )
    )

    destination.write_bytes(downloaded_path.read_bytes())
    print(f"[phase1] Saved checkpoint to {destination}")
    return destination


def is_loadable_checkpoint(checkpoint_path: Path) -> bool:
    """Check whether PyTorch can deserialize a checkpoint without raising."""
    try:
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return True
    except Exception as exc:
        print(f"[phase1] Existing checkpoint is not loadable: {exc}")
        return False


def resolve_checkpoint_path(checkpoint: str | None) -> Path:
    """Return a usable checkpoint path, downloading the default if needed."""
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists() and is_loadable_checkpoint(checkpoint_path):
            return checkpoint_path

        print(f"[phase1] Checkpoint not found or invalid at {checkpoint_path}; downloading default Push-T checkpoint instead.")
        return download_checkpoint(checkpoint_path)

    default_dir = REPO_ROOT / "checkpoints"
    default_path = default_dir / DEFAULT_CHECKPOINT_NAME
    if default_path.exists() and is_loadable_checkpoint(default_path):
        return default_path

    print(f"[phase1] No checkpoint provided; downloading default Push-T checkpoint to {default_path}.")
    return download_checkpoint(default_path)


def load_checkpoint_if_available(predictor: MaskedSlotPredictor, checkpoint_path: Path, device: torch.device) -> tuple[list[str], dict[str, int]]:
    """Load the checkpoint weights into the local predictor and report key mismatches."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found after resolution: {checkpoint_path}")

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = infer_checkpoint_spec(checkpoint_obj)
    if isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
    elif hasattr(checkpoint_obj, "model") and hasattr(checkpoint_obj.model, "predictor"):
        state_dict = checkpoint_obj.model.predictor.state_dict()
    elif hasattr(checkpoint_obj, "predictor"):
        state_dict = checkpoint_obj.predictor.state_dict()
    elif hasattr(checkpoint_obj, "state_dict"):
        state_dict = checkpoint_obj.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint_obj)}")

    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    predictor.to(device)

    if missing:
        print(f"[phase1] Note: checkpoint load skipped these keys because they do not map cleanly to the local predictor: {missing}")
    if unexpected:
        print(f"[phase1] Note: checkpoint load found extra keys that the smoke-test predictor does not use: {unexpected}")

    print(f"[phase1] Checkpoint loaded successfully from: {checkpoint_path}")
    return list(missing) + list(unexpected), spec


def load_slot_sample(slots_pkl: str | None, split: str, video_key: str | None, use_random_input: bool, args: argparse.Namespace) -> tuple[str, torch.Tensor]:
    """Load one slot tensor from disk or synthesize a random fixture for quick testing."""
    if use_random_input or not slots_pkl:
        sample = torch.randn(
            args.history_frames,
            args.num_slots,
            args.slot_dim,
            dtype=torch.float32,
        )
        print("[phase1] Using a synthetic random slot fixture because no slot pickle was requested.")
        return "random_fixture", sample

    slots_path = Path(slots_pkl)
    if not slots_path.exists():
        raise FileNotFoundError(f"Slot pickle not found: {slots_path}")

    with slots_path.open("rb") as handle:
        bank = pkl.load(handle)

    if split not in bank:
        raise KeyError(f"Split '{split}' not found in slot pickle. Available splits: {list(bank.keys())}")

    split_bank = bank[split]
    if not split_bank:
        raise ValueError(f"Split '{split}' is empty in {slots_path}")

    if video_key is None:
        video_key = next(iter(split_bank.keys()))

    if video_key not in split_bank:
        raise KeyError(f"Video key '{video_key}' not found in split '{split}'. Available keys: {list(split_bank.keys())[:10]}")

    sample = torch.as_tensor(split_bank[video_key], dtype=torch.float32)
    print(f"[phase1] Loaded slot sample from pickle: split={split}, video_key={video_key}")
    return video_key, sample


def summarize_tensor(tensor: torch.Tensor) -> dict[str, object]:
    """Convert a tensor into a JSON-friendly summary for the run report."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def resolve_device(device_name: str) -> torch.device:
    """Translate the CLI device argument into a concrete torch.device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def report_pass(step: str, detail: str, meaning: str) -> None:
    """Print a consistent PASS line plus a short plain-English meaning."""
    print(f"[phase1][PASS] {step} did {detail}, meaning {meaning}")


def parse_args() -> argparse.Namespace:
    """Parse the CLI flags for the local-only phase 1 smoke test."""
    parser = argparse.ArgumentParser(description="Phase 1 local-only Push-T C-JEPA smoke test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional C-JEPA checkpoint path")
    parser.add_argument("--slots-pkl", type=str, default=None, help="Optional pre-extracted Push-T slot pickle")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Slot split to sample from")
    parser.add_argument("--video-key", type=str, default=None, help="Optional video key inside the split")
    parser.add_argument("--use-random-input", action="store_true", help="Use a synthetic fixture instead of a pickle sample")
    parser.add_argument("--output-json", type=str, default="outputs/push_t_phase1_local_smoke.json", help="Where to write the run summary")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model initialization and fixture input")
    parser.add_argument("--num-slots", type=int, default=DEFAULT_NUM_SLOTS, help="Number of slots per frame")
    parser.add_argument("--slot-dim", type=int, default=DEFAULT_SLOT_DIM, help="Slot embedding dimension")
    parser.add_argument("--history-frames", type=int, default=DEFAULT_HISTORY_FRAMES, help="Number of observed history frames")
    parser.add_argument("--pred-frames", type=int, default=DEFAULT_PRED_FRAMES, help="Number of future frames to predict")
    parser.add_argument("--num-masked-slots", type=int, default=DEFAULT_NUM_MASKED_SLOTS, help="Number of masked slots")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Transformer depth")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS, help="Attention heads")
    parser.add_argument("--dim-head", type=int, default=DEFAULT_DIM_HEAD, help="Per-head attention dimension")
    parser.add_argument("--mlp-dim", type=int, default=DEFAULT_MLP_DIM, help="Transformer MLP dimension")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Transformer dropout")
    return parser.parse_args()


def main() -> int:
    """Run the full smoke test: resolve checkpoint, load a sample, and write a summary."""
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)

    # Resolve the checkpoint first so the runner can mirror its trained dimensions.
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_spec = infer_checkpoint_spec(checkpoint_obj)

    # Use the checkpoint's own dimensions so the smoke test matches the trained model.
    args.num_slots = checkpoint_spec["num_slots"]
    args.slot_dim = checkpoint_spec["slot_dim"]
    args.history_frames = checkpoint_spec["history_frames"]
    args.pred_frames = checkpoint_spec["pred_frames"]

    print(
        "[phase1] Model shape inferred from checkpoint: "
        f"num_slots={args.num_slots}, slot_dim={args.slot_dim}, "
        f"history_frames={args.history_frames}, pred_frames={args.pred_frames}"
    )
    report_pass(
        "checkpoint shape",
        f"match num_slots={args.num_slots}, slot_dim={args.slot_dim}, history_frames={args.history_frames}, pred_frames={args.pred_frames}",
        "the loaded checkpoint metadata matches the instantiated predictor configuration",
    )

    # Instantiate the local predictor with the same shape the checkpoint was trained on.
    predictor = build_predictor(args).to(device)
    predictor.eval()
    report_pass(
        "predictor build",
        f"initialize the local predictor on device={device}",
        "the predictor module is constructed and placed on the target compute device",
    )

    # Reuse the already-loaded checkpoint object for the weight transfer.
    if isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
    elif hasattr(checkpoint_obj, "model") and hasattr(checkpoint_obj.model, "predictor"):
        state_dict = checkpoint_obj.model.predictor.state_dict()
    elif hasattr(checkpoint_obj, "predictor"):
        state_dict = checkpoint_obj.predictor.state_dict()
    elif hasattr(checkpoint_obj, "state_dict"):
        state_dict = checkpoint_obj.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint_obj)}")

    # Load weights non-strictly so the smoke test can tolerate small wrapper differences.
    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    predictor.to(device)

    if missing:
        print(f"[phase1] Load report: skipped keys = {missing}")
    if unexpected:
        print(f"[phase1] Load report: extra keys in checkpoint = {unexpected}")

    report_pass(
        "checkpoint load",
        f"transfer the saved weights from {checkpoint_path}",
        "the serialized predictor parameters were deserialized into the local module",
    )

    # Pull one slot sequence from disk, or synthesize a fixture when --use-random-input is set.
    sample_name, sample = load_slot_sample(
        args.slots_pkl,
        args.split,
        args.video_key,
        args.use_random_input,
        args,
    )

    if sample.ndim != 3:
        raise ValueError(f"Expected slot tensor with shape [T, S, D], got {tuple(sample.shape)}")
    if sample.shape[0] < args.history_frames:
        raise ValueError(
            f"Sample has only {sample.shape[0]} frames, but history_frames={args.history_frames} requires more."
        )
    if sample.shape[1] != args.num_slots:
        raise ValueError(f"Sample slot count {sample.shape[1]} does not match num_slots={args.num_slots}")
    if sample.shape[2] != args.slot_dim:
        raise ValueError(f"Sample slot dim {sample.shape[2]} does not match slot_dim={args.slot_dim}")

    report_pass(
        "sample validation",
        f"confirm slot tensor shape [T, S, D] with T>={args.history_frames}, S={args.num_slots}, D={args.slot_dim}",
        "the input tensor satisfies the expected [T, S, D] slot layout and dimension constraints",
    )

    # Feed only the observed history into inference; the model predicts the future slots.
    history = sample[: args.history_frames].unsqueeze(0).to(device)
    print(
        "[phase1] Running inference on observed history only: "
        f"input_shape={tuple(history.shape)} -> predicting {args.pred_frames} future frame(s)"
    )

    with torch.no_grad():
        future = predictor.inference(history)

    report_pass(
        "inference",
        f"produce predicted tensor with shape {tuple(future.shape)} from input {tuple(history.shape)}",
        "the forward pass completed and returned a future slot tensor with the expected batch, time, slot, and embedding axes",
    )

    # Write a small JSON artifact so the run is easy to inspect or automate later.
    summary = {
        "phase": 1,
        "mode": "local_python",
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "sample": sample_name,
        "input": summarize_tensor(history),
        "output": summarize_tensor(future),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[phase1] Smoke test finished: checkpoint loaded, sample validated, inference ran, summary written.")
    print(f"[phase1] Sample used: {sample_name}")
    print(f"[phase1] Input tensor shape: {tuple(history.shape)}")
    print(f"[phase1] Predicted tensor shape: {tuple(future.shape)}")
    print(f"[phase1] JSON summary saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
