from huggingface_hub import hf_hub_download
import os
os.makedirs("checkpoints", exist_ok=True)

files = [
    "pusht_videosaur_model.ckpt",
    "pusht_videosaur_slots.pkl",
    "pusht_expert_action_meta.pkl",
    "pusht_expert_proprio_meta.pkl",
    "pusht_expert_state_meta.pkl",
]
for f in files:
    hf_hub_download("HazelNam/CJEPA", f, local_dir="checkpoints")
    print("Downloaded:", f)
