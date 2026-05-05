# Installation & Run Guide

## Prerequisites

| Requirement | Version (tested) | Notes |
|---|---|---|
| Python | 3.9–3.11 | venv at `.venv/` — avoid system Python 3.13 |
| ffmpeg | any | Must be on system PATH |
| CUDA | 12.x | For GPU training; CPU works but is ~10× slower |
| Git | any | Only if cloning third-party deps |

---

## 1. Clone the repo

```bash
git clone <repo-url>
cd soqqcjepa
```

---

## 2. Create the virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

---

## 3. Install Python dependencies

```bash
pip install stable-pretraining
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install lightning pytorch-lightning
pip install timm einops
pip install huggingface_hub datasets
pip install seaborn webdataset swig av accelerate tensorboard tensorboardX hickle pycocotools wget gdown
```

> For CPU-only (no GPU): replace `cu124` with `cpu` in the torch install URL.

---

## 4. Fix missing `stable_worldmodel.data` module

The repo bundles a partial `stable_worldmodel` that is missing the `data` submodule. Create stubs so imports succeed:

```powershell
mkdir stable_worldmodel\data
Write-Output "" > stable_worldmodel\data\__init__.py
Write-Output "from torch.utils.data import Dataset`n`nclass VideoDataset(Dataset):`n    def __init__(self, *args, **kwargs):`n        raise NotImplementedError('Install full stable-worldmodel from GitHub for CLEVRER support')" > stable_worldmodel\data\dataset.py
```

> Alternative: clone the full `stable-worldmodel` from GitHub into `src/third_party/` and `pip install -e .`, but the stubs above are sufficient for Push-T training.

---

## 5. Download data & checkpoints

Download from [HazelNam/CJEPA](https://huggingface.co/HazelNam/CJEPA) into `checkpoints/`:

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
os.makedirs("checkpoints", exist_ok=True)

files = [
    "pusht_videosaur_model.ckpt",          # VideoSAUR encoder
    "pusht_videosaur_slots.pkl",           # Pre-extracted slots (~4.7 GB)
    "pusht_expert_action_meta.pkl",        # Action metadata
    "pusht_expert_proprio_meta.pkl",       # Proprio metadata
    "pusht_expert_state_meta.pkl",         # State metadata
]
for f in files:
    hf_hub_download("HazelNam/CJEPA", f, local_dir="checkpoints")
    print("Downloaded:", f)
EOF
```

Expected layout:
```
checkpoints/
├── pusht_videosaur_model.ckpt           # VideoSAUR encoder (slot extractor)
├── pusht_videosaur_slots.pkl            # Pre-extracted slots
├── pusht_expert_action_meta.pkl         # Actions
├── pusht_expert_proprio_meta.pkl        # Proprioception
└── pusht_expert_state_meta.pkl          # State
```

---

## 6. Install ffmpeg (system dependency)

**Windows:** Download from https://www.gyan.dev/ffmpeg/builds/ and add `bin/` to PATH.

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

Verify: `ffmpeg -version`

---

## 7. Training (Push-T C-JEPA from pre-extracted slots)

### 7a. Patch VideoSAUR checkpoint loading (CPU-safe)

If training on CPU or loading a checkpoint saved on a different device, patch the `torch.load` call:

```powershell
python -c "(open('src/third_party/videosaur/videosaur/models.py').read().replace('checkpoint = torch.load(checkpoint_path)', 'checkpoint = torch.load(checkpoint_path, map_location=''cpu'')'))" | Set-Content -Path "src/third_party/videosaur/videosaur/models.py"
```

### 7b. Patch stable_pretraining for Windows (signal fix)

```powershell
python -c "(open('.venv/Lib/site-packages/stable_pretraining/manager.py').read().replace('    logging.info(f\"\\t\\t- SIGUSR1: `{signal.getsignal(signal.SIGUSR1)}`\")', '    if hasattr(signal, ''SIGUSR1''):`n        logging.info(f\"\\t\\t- SIGUSR1: `{signal.getsignal(signal.SIGUSR1)}`\")').replace('    logging.info(f\"\\t\\t- SIGUSR2: `{signal.getsignal(signal.SIGUSR2)}`\")', '    if hasattr(signal, ''SIGUSR2''):`n        logging.info(f\"\\t\\t- SIGUSR2: `{signal.getsignal(signal.SIGUSR2)}`\")').replace('    logging.info(f\"\\t\\t- SIGCONT: `{signal.getsignal(signal.SIGCONT)}`\")', '    if hasattr(signal, ''SIGCONT''):`n        logging.info(f\"\\t\\t- SIGCONT: `{signal.getsignal(signal.SIGCONT)}`\")'))" | Set-Content -Path ".venv/Lib/site-packages/stable_pretraining/manager.py"
```

### 7c. Run training

```powershell
$env:PYTHONPATH = "$pwd"
python src/train/train_causalwm_from_pusht_slot.py `
    trainer.max_epochs=100 `
    batch_size=128 `
    num_workers=4 `
    wandb.enable=false `
    predictor.heads=4 `
    model.load_weights=checkpoints/pusht_videosaur_model.ckpt `
    embedding_dir=checkpoints/pusht_videosaur_slots.pkl `
    action_dir=checkpoints/pusht_expert_action_meta.pkl `
    proprio_dir=checkpoints/pusht_expert_proprio_meta.pkl `
    state_dir=checkpoints/pusht_expert_state_meta.pkl
```

**Why `predictor.heads=4`?** The default config sets `proprio_embed_dim=10` and `action_embed_dim=10`, giving a total embedding dimension of `128 + 10 + 10 = 148`. The default `heads=16` fails because 148 is not divisible by 16. Valid head counts for 148 are 1, 2, or 4.

> For a quick smoke test, use `trainer.max_epochs=1`.

### 7d. Output

- **Per epoch**: `~\.stable_worldmodel\causal_world_model_epoch_{N}_object.ckpt`
- **Final**: `~\.stable_worldmodel\causal_world_model_object.ckpt`

---

## 8. Verify installation — Phase 1 smoke test (inference only)

```bash
cd scripts/pusht
python phase1_local_smoke.py --use-random-input
```

Expected: all `[phase1][PASS]` lines, exit 0.

---

## 9. Run Phase 2 — real Push-T frames (inference only)

### 9a. Extract frames from the public lerobot/pusht dataset

```bash
# Downloads ~6.9 MB once, cached at ~/.cache/lerobot_pusht/
python scripts/pusht/extract_lerobot_pusht_frames.py `
    --start-frame 500 --stride 10 `
    --out-dir scripts/pusht/sample_frames/pusht_dynamic_clip
```

### 9b. Run the full pipeline

```bash
python scripts/pusht/phase2_client_integration_smoke.py `
    --frames-dir scripts/pusht/sample_frames/pusht_dynamic_clip
```

Expected output includes latent evaluation metrics:
- `direct_mse` — predictor MSE vs true future slots
- `hungarian_mse` — permutation-invariant slot MSE  
- `repeat-last baseline MSE` — trivial baseline for comparison
- JSON summary written to `outputs/push_t_phase2_frame_sequence.json`

---

## Known issues

| Issue | Workaround |
|---|---|
| `ModuleNotFoundError: stable_worldmodel.data` | Create stub `stable_worldmodel/data/dataset.py` (Step 4) or clone full package |
| `RuntimeError: Attempting to deserialize object on CUDA device` | Patch `torch.load` to use `map_location='cpu'` (Step 7a) |
| `AttributeError: module 'signal' has no attribute 'SIGUSR1'` | Patch `stable_pretraining/manager.py` for Windows (Step 7b) |
| `AssertionError: embed_dim must be divisible by num_heads` | Override `predictor.heads=4` on CLI (148 ÷ 4 = 37) |
| `webdataset` import error in VideoSAUR | Already patched in `src/third_party/videosaur/videosaur/data/__init__.py` |
| `seaborn`/`cv2` import error in VideoSAUR | Already patched in `src/third_party/videosaur/videosaur/visualizations.py` |
| Released checkpoint has 6 slots, paper says 4 | The released `pusht_videosaur_1_epoch_30_object.ckpt` is an older run. Training from scratch with the repo config uses 4 slots correctly. |
| Original Push-T expert dataset (Google Drive) | Requires access from authors; use `lerobot/pusht` (HuggingFace) as substitute |

---

## Training notes

- **Dataset size:** 18,685 train videos + 21 val videos (~125 frames each)
- **Training samples:** ~1.9M windows (stride 1)
- **Batches per epoch:** ~14,900 at `batch_size=128`
- **Epoch time:** ~20 minutes on RTX 4070 Laptop GPU at batch_size=128
- **Checkpoint format:** The `_object.ckpt` files are full `torch.save(pl_module)` pickles compatible with the downstream loading code.
