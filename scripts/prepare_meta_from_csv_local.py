#!/usr/bin/env python3
"""
prepare_meta_from_csv_local.py

Read one or more CSV files from a directory containing per-frame metadata
and produce three pickles compatible with the training code:
  - pusht_expert_action_meta.pkl
  - pusht_expert_proprio_meta.pkl
  - pusht_expert_state_meta.pkl

The script uses an existing slots pickle (produced by the slot extractor)
to discover video keys and frame counts. It is forgiving about CSV column
names: it accepts grouped columns named like action_0, action_1... or a
single `action` column containing a python-list-like string ("[0.1, 0.2]").

Assumptions (reasonable defaults):
 - CSV files contain columns for episode index (episode or episode_idx) and
   frame/step index (step or step_idx).
 - Per-frame vectors are either split across several columns with a common
   prefix (action*, proprio*, state*) or contained in one column as a
   bracketed list string.

Usage example (PowerShell):
  $env:PYTHONPATH='C:\new\soqqcjepa'; .\.venv\Scripts\python.exe scripts/prepare_meta_from_csv_local.py --csv-dir testdata/mycsvs --slotpath checkpoints/my_own_slots.pkl --save-dir checkpoints

"""
import argparse
import os
import pickle as pkl
import glob
import csv
import ast
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Prepare action/proprio/state meta pickles from CSVs for local videos")
    p.add_argument("--csv-dir", required=True, help="Directory containing CSV files with per-frame metadata")
    p.add_argument("--slotpath", required=True, help="Path to slots pickle (produced by slot extractor)")
    p.add_argument("--save-dir", required=True, help="Directory to write the three meta pickles into")
    p.add_argument("--sep", default=",", help="CSV separator (default: ,)")
    return p.parse_args()


def load_slots(slotpath):
    with open(slotpath, "rb") as f:
        data = pkl.load(f)
    # normalize shapes: squeeze leading batch dim if present
    for split in list(data.keys()):
        for k, v in list(data[split].items()):
            arr = np.asarray(v)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            data[split][k] = arr
    return data


def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]
    episode_col = None
    step_col = None
    action_cols = []
    proprio_cols = []
    state_cols = []
    reward_cols = []
    for orig, l in zip(fieldnames, low):
        if l in ("episode", "episode_idx", "episodeid", "episode_id"):
            episode_col = orig
        elif l in ("step", "step_idx", "frame", "frame_idx", "stepidx", "step_index"):
            step_col = orig
        elif l.startswith("action"):
            action_cols.append(orig)
        # common local dataset conventions
        elif l.startswith("proprio"):
            proprio_cols.append(orig)
        elif l.startswith("agent_pos") or l.startswith("agent_rot") or l.startswith("agent_"):
            # agent position/rotation fields considered proprio
            proprio_cols.append(orig)
        elif l.startswith("block_pos") or l.startswith("block_vel") or l.startswith("goal_pos"):
            # block/goal related fields considered part of state
            state_cols.append(orig)
        elif l.startswith("state"):
            state_cols.append(orig)
        elif l.startswith("reward"):
            reward_cols.append(orig)
    # fallback guesses
    if episode_col is None:
        for orig, l in zip(fieldnames, low):
            if l.endswith("episode"):
                episode_col = orig
    if step_col is None:
        for orig, l in zip(fieldnames, low):
            if l.endswith("step") or l.endswith("frame"):
                step_col = orig
    return episode_col, step_col, action_cols, proprio_cols, state_cols, reward_cols


def parse_cell(cell):
    if cell is None:
        return None
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=np.float32)
    s = str(cell).strip()
    if s == "":
        return None
    # try literal eval (handles: "[0.1, 0.2]", "(1,2)")
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (int, float)):
            return np.array([float(val)], dtype=np.float32)
        if isinstance(val, (list, tuple)):
            return np.asarray(val, dtype=np.float32)
    except Exception:
        pass
    # try comma-separated numbers
    if "," in s:
        parts = [x.strip() for x in s.split(",") if x.strip()]
        try:
            return np.asarray([float(x) for x in parts], dtype=np.float32)
        except Exception:
            pass
    # fallback: single float
    try:
        return np.array([float(s)], dtype=np.float32)
    except Exception:
        return None


def ensure_width(arr, neww):
    # arr is np.ndarray shape (T, w) or None
    if arr is None:
        return None
    T, w = arr.shape
    if neww <= w:
        return arr
    out = np.zeros((T, neww), dtype=arr.dtype)
    out[:, :w] = arr
    return out


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    slots = load_slots(args.slotpath)

    # Prepare empty dicts for train/val keyed by the exact keys in slots
    train_action = {}
    train_proprio = {}
    train_state = {}
    train_reward = {}
    val_action = {}
    val_proprio = {}
    val_state = {}
    val_reward = {}

    for k, v in slots.get("train", {}).items():
        T = np.asarray(v).shape[0]
        train_action[k] = None
        train_proprio[k] = None
        train_state[k] = None
    train_reward[k] = None
    for k, v in slots.get("val", {}).items():
        T = np.asarray(v).shape[0]
        val_action[k] = None
        val_proprio[k] = None
        val_state[k] = None
    val_reward[k] = None

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {args.csv_dir}")

    for csvf in csv_files:
        with open(csvf, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=args.sep)
            fieldnames = reader.fieldnames or []
            ep_col, step_col, action_cols, proprio_cols, state_cols, reward_cols = detect_columns(fieldnames)
            # If grouped columns are not present, we may expect single columns named 'action','proprio','state'
            single_action = 'action' in fieldnames and not action_cols
            single_proprio = 'proprio' in fieldnames and not proprio_cols
            single_state = 'state' in fieldnames and not state_cols
            single_reward = 'reward' in fieldnames and not reward_cols

            for row in tqdm(reader, desc=f"Processing {os.path.basename(csvf)}"):
                # episode/frame
                if ep_col is None or step_col is None:
                    # try numeric columns fallback
                    keys = {c.lower(): c for c in fieldnames}
                    epcol = keys.get('episode') or keys.get('episode_idx')
                    stepcol = keys.get('step') or keys.get('step_idx') or keys.get('frame')
                else:
                    epcol = ep_col
                    stepcol = step_col

                try:
                    episode = int(float(row[epcol]))
                    frame = int(float(row[stepcol]))
                except Exception:
                    # skip rows that don't have episode/step info
                    continue

                key = f"{episode}_pixels.mp4"
                split = 'train' if key in train_action else ('val' if key in val_action else None)
                if split is None:
                    # try nearby episode indexes (common off-by-one from 1-based logs)
                    tried = False
                    for alt in (episode - 1, episode + 1):
                        if alt < 0:
                            continue
                        alt_key = f"{alt}_pixels.mp4"
                        if alt_key in train_action:
                            key = alt_key; split = 'train'; tried = True
                            print(f"Info: remapped episode {episode} -> {alt} for train key {key}")
                            break
                        if alt_key in val_action:
                            key = alt_key; split = 'val'; tried = True
                            print(f"Info: remapped episode {episode} -> {alt} for val key {key}")
                            break
                    if not tried:
                        # if there's exactly one train video, map everything to it (useful for single-video local data)
                        if len(train_action) == 1 and len(val_action) == 0:
                            key = next(iter(train_action.keys()))
                            split = 'train'
                            print(f"Info: mapped episode {episode} to single available train video {key}")
                        else:
                            # unknown video key, skip
                            print(f"Warning: skipping row with episode {episode} because no matching slot key found")
                            continue

                # helper to get vector from grouped or single columns
                def get_vec(cols, single_flag):
                    if cols:
                        parts = []
                        for c in cols:
                            parts.append(row.get(c, ""))
                        # parse each part (scalar) into float
                        parsed = []
                        for p in parts:
                            parsed_cell = parse_cell(p)
                            if parsed_cell is None:
                                parsed.append(0.0)
                            else:
                                # parsed_cell may be array even for scalar; take first
                                parsed.append(float(np.asarray(parsed_cell).ravel()[0]))
                        return np.asarray(parsed, dtype=np.float32)
                    if single_flag:
                        raw = row.get('action') if single_flag and 'action' in row else row.get('action')
                        return parse_cell(raw)
                    return None

                # action
                action_vec = None
                if action_cols:
                    action_vec = get_vec(action_cols, False)
                elif single_action:
                    action_vec = parse_cell(row.get('action'))
                else:
                    # try to find any column that contains '['
                    for c in fieldnames:
                        if '[' in (row.get(c, '') or '') and 'action' in c.lower():
                            action_vec = parse_cell(row.get(c))
                            break

                # proprio
                proprio_vec = None
                if proprio_cols:
                    proprio_vec = get_vec(proprio_cols, False)
                elif single_proprio:
                    proprio_vec = parse_cell(row.get('proprio'))

                # state
                state_vec = None
                if state_cols:
                    state_vec = get_vec(state_cols, False)
                elif single_state:
                    state_vec = parse_cell(row.get('state'))

                # reward
                reward_vec = None
                if reward_cols:
                    reward_vec = get_vec(reward_cols, False)
                elif single_reward:
                    reward_vec = parse_cell(row.get('reward'))

                # choose target dicts
                if split == 'train':
                    a_dict, p_dict, s_dict, r_dict = train_action, train_proprio, train_state, train_reward
                    max_frames = np.asarray(slots['train'][key]).shape[0]
                else:
                    a_dict, p_dict, s_dict, r_dict = val_action, val_proprio, val_state, val_reward
                    max_frames = np.asarray(slots['val'][key]).shape[0]

                # initialize arrays lazily with width inferred from first vector seen
                if action_vec is not None:
                    if a_dict[key] is None:
                        w = int(action_vec.size)
                        a_dict[key] = np.zeros((max_frames, w), dtype=np.float32)
                    if frame < a_dict[key].shape[0]:
                        if action_vec.size != a_dict[key].shape[1]:
                            a_dict[key] = ensure_width(a_dict[key], int(action_vec.size))
                        a_dict[key][frame, :action_vec.size] = action_vec

                if proprio_vec is not None:
                    if p_dict[key] is None:
                        w = int(proprio_vec.size)
                        p_dict[key] = np.zeros((max_frames, w), dtype=np.float32)
                    if frame < p_dict[key].shape[0]:
                        if proprio_vec.size != p_dict[key].shape[1]:
                            p_dict[key] = ensure_width(p_dict[key], int(proprio_vec.size))
                        p_dict[key][frame, :proprio_vec.size] = proprio_vec

                if state_vec is not None:
                    if s_dict[key] is None:
                        w = int(state_vec.size)
                        s_dict[key] = np.zeros((max_frames, w), dtype=np.float32)
                    if frame < s_dict[key].shape[0]:
                        if state_vec.size != s_dict[key].shape[1]:
                            s_dict[key] = ensure_width(s_dict[key], int(state_vec.size))
                        s_dict[key][frame, :state_vec.size] = state_vec

                if reward_vec is not None:
                    if r_dict[key] is None:
                        w = int(reward_vec.size)
                        r_dict[key] = np.zeros((max_frames, w), dtype=np.float32)
                    if frame < r_dict[key].shape[0]:
                        if reward_vec.size != r_dict[key].shape[1]:
                            r_dict[key] = ensure_width(r_dict[key], int(reward_vec.size))
                        r_dict[key][frame, :reward_vec.size] = reward_vec

    # Finalize: replace any None arrays with zeros of (T,1)
    def finalize_map(dmap, slots_split):
        out = {}
        for k, arr in dmap.items():
            if k not in slots_split:
                # CSV referenced a video key not present in the slots pickle; skip it
                print(f"Warning: skipping key {k} because it's not present in slots")
                continue
            T = np.asarray(slots_split[k]).shape[0]
            if arr is None:
                out[k] = np.zeros((T, 1), dtype=np.float32)
            else:
                out[k] = arr
        return out

    action_file = {"train": finalize_map(train_action, slots['train']), "val": finalize_map(val_action, slots['val'])}
    proprio_file = {"train": finalize_map(train_proprio, slots['train']), "val": finalize_map(val_proprio, slots['val'])}
    state_file = {"train": finalize_map(train_state, slots['train']), "val": finalize_map(val_state, slots['val'])}
    reward_file = {"train": finalize_map(train_reward, slots['train']), "val": finalize_map(val_reward, slots['val'])}

    # write meta pickles using a neutral "local" prefix (your data is not pusht)
    with open(os.path.join(args.save_dir, "local_action_meta.pkl"), "wb") as f:
        pkl.dump(action_file, f)
    with open(os.path.join(args.save_dir, "local_proprio_meta.pkl"), "wb") as f:
        pkl.dump(proprio_file, f)
    with open(os.path.join(args.save_dir, "local_state_meta.pkl"), "wb") as f:
        pkl.dump(state_file, f)
    with open(os.path.join(args.save_dir, "local_reward_meta.pkl"), "wb") as f:
        pkl.dump(reward_file, f)

    print("Saved meta pickles to", os.path.abspath(args.save_dir))


if __name__ == '__main__':
    main()
