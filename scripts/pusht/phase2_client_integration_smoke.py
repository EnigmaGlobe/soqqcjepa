"""Phase 2 frame-sequence Push-T C-JEPA pipeline smoke test.

This script proves the next step after the phase 1 local run:
- load a short ordered frame sequence from disk or a synthetic fixture
- run the video through a Push-T VideoSAUR checkpoint to get slots
- reshape the slots to match the Push-T world-model contract
- run the C-JEPA world model to predict the next latent state

The goal is not benchmark scoring. The goal is to show the full latent
pipeline from frame sequence -> VideoSAUR slots -> C-JEPA prediction in one Python main.

Original sources:
- VideoSAUR paper: https://arxiv.org/abs/2306.04829
- VideoSAUR repository: https://github.com/martius-lab/videosaur
- VideoSAUR project page: https://martius-lab.github.io/videosaur
- Push-T C-JEPA checkpoint source: https://huggingface.co/HazelNam/CJEPA
- Stable-Pretraining: https://galilai-group.github.io/stable-pretraining/
- Stable-WorldModel: https://galilai-group.github.io/stable-worldmodel/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torchvision.io import read_image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cjepa_predictor import MaskedSlotPredictor
from src.custom_codes.hungarian import hungarian_matching_loss_AP
from src.third_party.videosaur.videosaur import configuration, models
from src.third_party.videosaur.videosaur.data.transforms import build_inference_transform

DEFAULT_WORLD_MODEL_CHECKPOINT_NAME = "pusht_videosaur_1_epoch_30_object.ckpt"
DEFAULT_VIDEOSAUR_CHECKPOINT_NAME = "pusht_videosaur_model.ckpt"
DEFAULT_WORLD_MODEL_CHECKPOINT_URL = "https://huggingface.co/HazelNam/CJEPA/resolve/main/cjepa-ckpts/pusht_videosaur_1_epoch_30_object.ckpt"
DEFAULT_VIDEOSAUR_CHECKPOINT_URL = "https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_videosaur_model.ckpt"
DEFAULT_VIDEOSAUR_CONFIG = REPO_ROOT / "src" / "third_party" / "videosaur" / "configs" / "videosaur" / "pusht_dinov2_hf.yml"
DEFAULT_FRAMES_DIR = REPO_ROOT / "scripts" / "pusht" / "sample_frames" / "circle_to_square"
DEFAULT_IMAGE_SIZE = 196


def report_pass(step: str, detail: str, meaning: str) -> None:
    """Print a consistent PASS line for the phase 2 pipeline."""
    print(f"[phase2][PASS] {step} did {detail}, meaning {meaning}")


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


def summarize_scalar(value: torch.Tensor | float) -> float:
    """Convert a scalar tensor or Python number into a JSON-friendly float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def resolve_device(device_name: str) -> torch.device:
    """Translate the CLI device argument into a concrete torch.device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_or_download_checkpoint(destination: Path, repo_filename: str, url: str) -> Path:
    """Download a checkpoint from Hugging Face if the local path is missing or unusable."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            torch.load(destination, map_location="cpu", weights_only=False)
            return destination
        except Exception as exc:  # noqa: BLE001
            print(f"[phase2] Existing checkpoint is not loadable at {destination}: {exc}")

    cache_dir = destination.parent / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase2] Downloading checkpoint from Hugging Face: {url}")
    downloaded_path = Path(
        hf_hub_download(
            repo_id="HazelNam/CJEPA",
            filename=repo_filename,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            force_download=True,
        )
    )
    destination.write_bytes(downloaded_path.read_bytes())
    print(f"[phase2] Saved checkpoint to {destination}")
    return destination


def resolve_world_model_checkpoint(checkpoint: str | None) -> Path:
    """Return a usable Push-T C-JEPA checkpoint path."""
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            return checkpoint_path
        return load_or_download_checkpoint(checkpoint_path, DEFAULT_WORLD_MODEL_CHECKPOINT_NAME, DEFAULT_WORLD_MODEL_CHECKPOINT_URL)

    default_path = REPO_ROOT / "checkpoints" / DEFAULT_WORLD_MODEL_CHECKPOINT_NAME
    return load_or_download_checkpoint(default_path, DEFAULT_WORLD_MODEL_CHECKPOINT_NAME, DEFAULT_WORLD_MODEL_CHECKPOINT_URL)


def resolve_videosaur_checkpoint(checkpoint: str | None) -> Path:
    """Return a usable VideoSAUR checkpoint path."""
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            return checkpoint_path
        return load_or_download_checkpoint(checkpoint_path, DEFAULT_VIDEOSAUR_CHECKPOINT_NAME, DEFAULT_VIDEOSAUR_CHECKPOINT_URL)

    default_path = REPO_ROOT / "checkpoints" / DEFAULT_VIDEOSAUR_CHECKPOINT_NAME
    return load_or_download_checkpoint(default_path, DEFAULT_VIDEOSAUR_CHECKPOINT_NAME, DEFAULT_VIDEOSAUR_CHECKPOINT_URL)


def infer_checkpoint_spec(checkpoint_obj) -> dict[str, int]:
    """Read the world-model dimensions from the loaded checkpoint object."""
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


def build_world_predictor(checkpoint_path: Path, device: torch.device) -> tuple[MaskedSlotPredictor, dict[str, int]]:
    """Load checkpoint weights into a local C-JEPA predictor."""
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_spec = infer_checkpoint_spec(checkpoint_obj)

    predictor = MaskedSlotPredictor(
        num_slots=checkpoint_spec["num_slots"],
        slot_dim=checkpoint_spec["slot_dim"],
        history_frames=checkpoint_spec["history_frames"],
        pred_frames=checkpoint_spec["pred_frames"],
        num_masked_slots=2,
        seed=0,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
    ).to(device)
    predictor.eval()

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
    if missing:
        print(f"[phase2] World model skipped keys: {missing}")
    if unexpected:
        print(f"[phase2] World model extra keys: {unexpected}")

    report_pass(
        "world model load",
        f"transfer weights from {checkpoint_path}",
        "the C-JEPA predictor is ready to consume latent slots",
    )
    return predictor, checkpoint_spec


def load_videosaur_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    """Build and load the Push-T VideoSAUR model."""
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if hasattr(config, "globals") and hasattr(config.globals, "NUM_SLOTS"):
        try:
            model.initializer.n_slots = int(config.globals.NUM_SLOTS)
        except Exception:  # noqa: BLE001
            pass

    report_pass(
        "videosaur load",
        f"transfer weights from {checkpoint_path}",
        "the VideoSAUR slot extractor is ready to turn frames into latent objects",
    )
    return model, config


def generate_demo_frame(size: int, frame_index: int, frame_count: int) -> torch.Tensor:
    """Generate one frame of a simple circle-to-square morph for smoke testing."""
    axis = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    progress = 0.0 if frame_count <= 1 else frame_index / float(frame_count - 1)
    circle = (xx.pow(2) + yy.pow(2)) <= 0.45**2
    square = (xx.abs() <= 0.45) & (yy.abs() <= 0.45)
    blend = ((1.0 - progress) * circle.float() + progress * square.float()) >= 0.5

    image = torch.zeros(3, size, size, dtype=torch.uint8)
    image[0].fill_(25)
    image[1].fill_(25)
    image[2].fill_(25)
    image[:, blend] = torch.tensor([255, 255, 255], dtype=torch.uint8).unsqueeze(1)
    return image


def generate_demo_sequence(size: int, frame_count: int) -> tuple[list[str], torch.Tensor]:
    """Generate a short synthetic frame sequence for smoke testing."""
    frame_names = []
    frames = []
    for frame_index in range(frame_count):
        frame_names.append(f"synthetic_{frame_index + 1:02d}")
        frames.append(generate_demo_frame(size, frame_index, frame_count))
    return frame_names, torch.stack(frames, dim=0)


def load_frame_sequence(frames_dir: str | None, use_random_input: bool, image_size: int, demo_frame_count: int) -> tuple[str, list[str], torch.Tensor]:
    """Load an ordered frame sequence from disk or synthesize a demo sequence."""
    if use_random_input or not frames_dir:
        frame_names, frames = generate_demo_sequence(image_size, demo_frame_count)
        return "synthetic_circle_to_square", frame_names, frames

    path = Path(frames_dir)
    if not path.exists():
        raise FileNotFoundError(f"Frame directory not found: {path}")

    frame_paths = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not frame_paths:
        raise ValueError(f"No image frames found in: {path}")

    frame_names = [candidate.name for candidate in frame_paths]
    frames = [read_image(str(candidate)) for candidate in frame_paths]
    return path.name, frame_names, torch.stack(frames, dim=0)

def frames_to_video(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert a frame stack into a normalized video tensor."""
    if frames.dtype != torch.float32:
        frames = frames.float() / 255.0

    transform_cfg = OmegaConf.create({"dataset_type": "video", "input_size": image_size})
    video = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
    video = build_inference_transform(transform_cfg)(video)
    return video.permute(1, 0, 2, 3).unsqueeze(0)  # [B=1, T, C, H, W]


def extract_video_slots(videosaur_model, video: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run VideoSAUR and return the latent slot sequence."""
    inputs = {"video": video.to(device)}
    with torch.no_grad():
        outputs = videosaur_model(inputs)

    slots = outputs["processor"]["state"]
    report_pass(
        "videosaur inference",
        f"produce slot tensor with shape {tuple(slots.shape)} from input {tuple(video.shape)}",
        "the frame sequence has been converted into VideoSAUR latent slots",
    )
    return slots


def pad_slots(slots: torch.Tensor, target_slots: int) -> torch.Tensor:
    """Pad or trim slots so they match the world-model checkpoint contract."""
    current_slots = slots.shape[2]
    if current_slots == target_slots:
        return slots

    if current_slots > target_slots:
        return slots[:, :, :target_slots]

    pad = torch.zeros(
        slots.shape[0],
        slots.shape[1],
        target_slots - current_slots,
        slots.shape[3],
        device=slots.device,
        dtype=slots.dtype,
    )
    return torch.cat([slots, pad], dim=2)


def repeat_history(slots: torch.Tensor, history_frames: int) -> torch.Tensor:
    """Expand a one-frame latent sequence into the history length expected by the world model."""
    if slots.shape[1] == history_frames:
        return slots
    if slots.shape[1] == 1:
        return slots.repeat(1, history_frames, 1, 1)
    if slots.shape[1] > history_frames:
        return slots[:, -history_frames:]
    repeat_count = history_frames // slots.shape[1] + int(history_frames % slots.shape[1] != 0)
    repeated = slots.repeat(1, repeat_count, 1, 1)
    return repeated[:, :history_frames]


def build_eval_windows(
    slots: torch.Tensor,
    history_frames: int,
    pred_frames: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build all valid history/target windows for latent next-step evaluation."""
    total_frames = slots.shape[1]
    required_frames = history_frames + pred_frames
    if total_frames < required_frames:
        return None

    histories = []
    targets = []
    for start_index in range(total_frames - required_frames + 1):
        histories.append(slots[:, start_index:start_index + history_frames])
        targets.append(slots[:, start_index + history_frames:start_index + required_frames])

    return torch.cat(histories, dim=0), torch.cat(targets, dim=0)


def evaluate_future_prediction(predicted_future: torch.Tensor, true_future: torch.Tensor, history: torch.Tensor) -> dict[str, object]:
    """Compute the latent-space metrics this repo uses to judge future-slot prediction."""
    direct_mse = F.mse_loss(predicted_future, true_future)
    repeat_last_baseline = F.mse_loss(history[:, -predicted_future.shape[1]:], true_future)
    hungarian_mse = hungarian_matching_loss_AP(predicted_future, true_future, cost_type="mse", reduction="mean")["pixels_loss"]
    hungarian_cosine = hungarian_matching_loss_AP(predicted_future, true_future, cost_type="cosine", reduction="mean")["pixels_loss"]

    return {
        "window_count": int(predicted_future.shape[0]),
        "pred_frames": int(predicted_future.shape[1]),
        "direct_mse": summarize_scalar(direct_mse),
        "hungarian_mse": summarize_scalar(hungarian_mse),
        "hungarian_cosine": summarize_scalar(hungarian_cosine),
        "repeat_last_frame_mse": summarize_scalar(repeat_last_baseline),
        "beats_repeat_last_baseline": bool(direct_mse < repeat_last_baseline),
    }


def parse_args() -> argparse.Namespace:
    """Parse the CLI flags for the phase 2 frame-sequence smoke test."""
    parser = argparse.ArgumentParser(description="Phase 2 Push-T frame-sequence pipeline smoke test")
    parser.add_argument("--frames-dir", type=str, default=str(DEFAULT_FRAMES_DIR), help="Directory containing an ordered frame sequence")
    parser.add_argument("--use-random-input", action="store_true", help="Use a synthetic demo frame sequence instead of files")
    parser.add_argument("--output-json", type=str, default="outputs/push_t_phase2_frame_sequence.json", help="Where to write the run summary")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the synthetic image fixture")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Resize target for the VideoSAUR input frames")
    parser.add_argument("--demo-frame-count", type=int, default=5, help="Number of synthetic frames to generate when no folder is used")
    parser.add_argument("--videosaur-config", type=str, default=str(DEFAULT_VIDEOSAUR_CONFIG), help="VideoSAUR model config path")
    parser.add_argument("--videosaur-checkpoint", type=str, default=None, help="Optional VideoSAUR checkpoint path")
    parser.add_argument("--world-model-checkpoint", type=str, default=None, help="Optional Push-T C-JEPA checkpoint path")
    return parser.parse_args()


def main() -> int:
    """Run frame sequence -> VideoSAUR slots -> C-JEPA prediction and save a JSON summary."""
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)

    videosaur_checkpoint = resolve_videosaur_checkpoint(args.videosaur_checkpoint)
    world_model_checkpoint = resolve_world_model_checkpoint(args.world_model_checkpoint)

    videosaur_model, videosaur_config = load_videosaur_model(Path(args.videosaur_config), videosaur_checkpoint, device)
    world_predictor, world_spec = build_world_predictor(world_model_checkpoint, device)

    sequence_name, frame_names, frames = load_frame_sequence(args.frames_dir, args.use_random_input, args.image_size, args.demo_frame_count)
    video = frames_to_video(frames, args.image_size)
    report_pass(
        "frame load",
        f"load frame sequence {sequence_name} with {len(frame_names)} frames and convert it to video shape {tuple(video.shape)}",
        "the pipeline starts from an actual ordered frame sequence instead of a slot fixture",
    )

    slot_sequence = extract_video_slots(videosaur_model, video, device)
    slot_sequence = pad_slots(slot_sequence, world_spec["num_slots"])
    report_pass(
        "slot padding",
        f"match VideoSAUR slots to world-model slot count {world_spec['num_slots']} with shape {tuple(slot_sequence.shape)}",
        "the extracted latent slots now fit the world-model checkpoint contract",
    )

    eval_windows = build_eval_windows(slot_sequence, world_spec["history_frames"], world_spec["pred_frames"])
    if eval_windows is not None:
        world_history, true_future = eval_windows
        history_meaning = "the ordered frames provide real history/target windows for latent evaluation"
    else:
        world_history = repeat_history(slot_sequence, world_spec["history_frames"])
        true_future = None
        history_meaning = "the frame sequence has been padded or trimmed into the temporal input the predictor expects"

    report_pass(
        "history shaping",
        f"expand slot history to shape {tuple(world_history.shape)} for the world model",
        history_meaning,
    )

    with torch.no_grad():
        future = world_predictor.inference(world_history)

    report_pass(
        "world prediction",
        f"produce predicted latent tensor with shape {tuple(future.shape)} from world input {tuple(world_history.shape)}",
        "the full image -> slots -> prediction pipeline completed successfully",
    )

    evaluation = None
    if true_future is not None:
        evaluation = evaluate_future_prediction(future, true_future, world_history)
        report_pass(
            "latent evaluation",
            (
                f"score predicted future slots against true future slots over {evaluation['window_count']} windows "
                f"with direct_mse={evaluation['direct_mse']:.6f} and hungarian_mse={evaluation['hungarian_mse']:.6f}"
            ),
            "the phase 2 run now reports the same kind of latent-space accuracy signals used by the project",
        )

    summary = {
        "phase": 2,
        "mode": "frame_sequence_pipeline",
        "device": str(device),
        "sequence": sequence_name,
        "frame_names": frame_names,
        "frame_count": len(frame_names),
        "frame_shape": list(frames.shape),
        "video_shape": list(video.shape),
        "videosaur_checkpoint": str(videosaur_checkpoint),
        "videosaur_config": str(args.videosaur_config),
        "videosaur_slots": summarize_tensor(slot_sequence),
        "world_model_checkpoint": str(world_model_checkpoint),
        "world_model_spec": world_spec,
        "world_history_shape": list(world_history.shape),
        "world_prediction": summarize_tensor(future),
        "true_future": summarize_tensor(true_future) if true_future is not None else None,
        "evaluation": evaluation,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[phase2] Pipeline finished: frame sequence -> VideoSAUR slots -> C-JEPA prediction.")
    print(f"[phase2] Frame sequence used: {sequence_name}")
    print(f"[phase2] Frame count: {len(frame_names)}")
    print(f"[phase2] VideoSAUR slots shape: {tuple(slot_sequence.shape)}")
    print(f"[phase2] World history shape: {tuple(world_history.shape)}")
    print(f"[phase2] Predicted tensor shape: {tuple(future.shape)}")
    if evaluation is not None:
        print(f"[phase2] Direct future-slot MSE: {evaluation['direct_mse']:.6f}")
        print(f"[phase2] Hungarian future-slot MSE: {evaluation['hungarian_mse']:.6f}")
        print(f"[phase2] Hungarian future-slot cosine cost: {evaluation['hungarian_cosine']:.6f}")
        print(f"[phase2] Repeat-last baseline MSE: {evaluation['repeat_last_frame_mse']:.6f}")
        print(f"[phase2] Predictor beats repeat-last baseline: {evaluation['beats_repeat_last_baseline']}")
    print(f"[phase2] JSON summary saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
