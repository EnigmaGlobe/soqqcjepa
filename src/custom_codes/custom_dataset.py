import numpy as np
from stable_worldmodel.data.dataset import VideoDataset, Dataset
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from torch.utils.data import Dataset, DataLoader
from loguru import logger as logging

import pickle as pkl


class ClevrerVideoDataset(VideoDataset):
    """
    Custom VideoDataset for CLEVRER with episode index offset support.

    This class extends VideoDataset to add an idx_offset parameters  that shifts
    all episode indices by a constant value. 
    """

    def __init__(self, name, *args, idx_offset=0, **kwargs):
        # Call parent VideoDataset.__init__
        super().__init__(name, *args, **kwargs)

        # Store the offset
        self.idx_offset =idx_offset


    def __repr__(self):
        return (
            f"ClevrerVideoDataset(name='{self.dataset}', "
            f"num_episodes={len(self.episodes)}, "
            f"idx_offset={self.idx_offset}, "
            f"frameskip={self.frameskip}, "
            f"num_steps={self.num_steps})"
        )

    def __getitem__(self, index):
        episode = self.idx_to_episode[index]
        episode_indices = self.episode_indices[episode+self.idx_offset]
        offset = index - self.episode_starts[episode]

        # determine clip bounds
        start = offset if not self.complete_traj else 0
        stop = start + self.clip_len if not self.complete_traj else len(self.episode_indices[episode+self.idx_offset])
        step_slice = episode_indices[start:stop]
        steps = self.dataset[step_slice]

        for col, data in steps.items():
            if col == "action":
                continue

            data = data[:: self.frameskip]
            steps[col] = data

            if col in self.decode_columns:
                steps[col] = self.decode(steps["data_dir"], steps[col], start=start, end=stop)

        if self.transform:
            steps = self.transform(steps)

        # stack frames
        for col in self.decode_columns:
            if col not in steps:
                continue
            steps[col] = torch.stack(steps[col])

        # reshape action
        if "action" in steps:
            act_shape = self.num_steps if not self.complete_traj else len(self.episode_indices[episode+self.idx_offset])
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps
    


# ============================================================================
# Dataset for Pre-extracted Slot Representations
# ============================================================================
class PushTSlotDataset(Dataset):
    """
    Dataset for pre-extracted slot representations from PushT.
    
    This class mirrors the behavior of swm.data.VideoDataset to ensure
    identical data processing. Key behaviors:
    - Window stride of 1 (not frameskip) for sample indices
    - Action is reshaped to (T, action_dim * frameskip) by VideoDataset
    - Normalization uses mean/std without clamping (same as WrapTorchTransform)
    - nan_to_num is only applied in forward pass, not in dataset
    
    Each sample contains:
    - pixels_embed: Pre-extracted slot embeddings (T, num_slots, slot_dim)
    - action: Action sequence (T, action_dim * frameskip)
    - proprio: Proprioception sequence (T, proprio_dim)
    - state: State sequence (T, state_dim) [optional, for evaluation]
    
    Args:
        slot_data: Dict mapping video_id to slot embeddings
        split: 'train' or 'val'
        history_size: Number of history frames
        num_preds: Number of future frames to predict
        action_dir: Path to action pickle file
        proprio_dir: Path to proprioception pickle file
        state_dir: Path to state pickle file (optional)
        frameskip: Frame skip factor (affects action reshaping)
        seed: Random seed for sampling
    """
    
    def __init__(
        self,
        slot_data: dict,
        split: str,
        history_size: int,
        num_preds: int,
        action_dir: str,
        proprio_dir: str,
        state_dir: str = None,
        frameskip: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.slot_data = slot_data
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip
        self.n_steps = history_size + num_preds
        self.seed = seed
        
        # Load action and proprio data
        with open(action_dir, "rb") as f:
            action_data = pkl.load(f)
        self.action_data = action_data[split]
        
        with open(proprio_dir, "rb") as f:
            proprio_data = pkl.load(f)
        self.proprio_data = proprio_data[split]
        
        # State is optional (used for evaluation)
        self.state_data = None
        if state_dir is not None:
            with open(state_dir, "rb") as f:
                state_data = pkl.load(f)
            self.state_data = state_data[split]
        
        # Build index: list of (video_id, start_frame) tuples
        self.samples = self._build_sample_index()
        
        # Compute normalization statistics (matching WrapTorchTransform behavior)
        self._compute_normalization_stats()
        
        logging.info(f"[{split}] Created dataset with {len(self.samples)} samples from {len(self.slot_data)} videos")
    
    def _build_sample_index(self):
        """
        Build list of valid (video_id, start_frame) samples.
        
        Matches VideoDataset behavior: stride of 1, not frameskip.
        VideoDataset uses: episode_max_end = max(0, len(ep) - clip_len + 1)
        and iterates over all start positions with stride 1.
        """
        samples = []
        clip_len = self.n_steps * self.frameskip
        
        for video_id, slots in self.slot_data.items():
            num_frames = slots.shape[0]
            # max_start is inclusive, so we can start at positions 0 to max_start
            max_start = num_frames - clip_len
            
            if max_start < 0:
                continue
            
            # Stride 1 matching VideoDataset behavior
            for start_idx in range(0, max_start + 1):
                samples.append((video_id, start_idx))
        
        return samples
    
    def _compute_normalization_stats(self):
        """
        Compute mean and std for action and proprio normalization.
        
        Matches WrapTorchTransform(norm_col_transform(dataset, col)) behavior:
        - Computes stats over the RESHAPED action column (T, action_dim * frameskip)
        - No clamping of std (WrapTorchTransform doesn't clamp)
        - Uses tensor mean/std with unsqueeze(0)
        
        Note: VideoDataset reshapes action to (T, -1) before transform is applied.
        """
        # Collect all actions and proprios in their RESHAPED form
        # This matches how VideoDataset provides data to the transform
        all_actions = []
        all_proprios = []
        
        for video_id in self.action_data.keys():
            action_raw = self.action_data[video_id]  # (num_frames, action_dim)
            # Reshape to match VideoDataset's reshape: (T, action_dim * frameskip)
            # VideoDataset does: steps["action"].reshape(act_shape, -1)
            # where act_shape = num_steps and the raw actions are clip_len = n_steps * frameskip
            # So each T gets frameskip consecutive actions flattened
            num_frames = action_raw.shape[0]
            clip_len = self.n_steps * self.frameskip
            
            # Iterate over all possible clips (stride 1, matching _build_sample_index)
            for start_idx in range(0, num_frames - clip_len + 1):
                # Get clip_len consecutive raw actions
                action_clip = action_raw[start_idx:start_idx + clip_len]  # (clip_len, action_dim)
                # Reshape to (n_steps, action_dim * frameskip) - matching VideoDataset
                action_reshaped = action_clip.reshape(self.n_steps, -1)
                all_actions.append(action_reshaped)
        
        for video_id in self.proprio_data.keys():
            proprio_raw = self.proprio_data[video_id]  # (num_frames, proprio_dim)
            num_frames = proprio_raw.shape[0]
            clip_len = self.n_steps * self.frameskip
            
            for start_idx in range(0, num_frames - clip_len + 1):
                # Get frames with frameskip (matching VideoDataset: data[::frameskip])
                frame_indices = [start_idx + i * self.frameskip for i in range(self.n_steps)]
                if frame_indices[-1] < num_frames:
                    proprio_clip = proprio_raw[frame_indices]  # (n_steps, proprio_dim)
                    all_proprios.append(proprio_clip)
        
        # Stack and compute stats matching norm_col_transform:
        # data.mean(0).unsqueeze(0), data.std(0).unsqueeze(0)
        if len(all_actions) > 0:
            all_actions = torch.from_numpy(np.concatenate(all_actions, axis=0)).float()  # (N*T, action_dim*frameskip)
            self.action_mean = all_actions.mean(0).unsqueeze(0)  # (1, action_dim * frameskip)
            self.action_std = all_actions.std(0).unsqueeze(0)    # (1, action_dim * frameskip)
        else:
            # Fallback: infer action width from an example in action_data or default to 1
            example_shape = None
            for v in self.action_data.values():
                example = np.asarray(v)
                if example.size > 0:
                    example_shape = example.shape
                    break
            if example_shape is not None:
                action_dim = example_shape[1]
                width = int(action_dim * self.frameskip)
            else:
                width = 1
            logging.warning("No action clips found for normalization; falling back to zeros/ones of width %s", width)
            self.action_mean = torch.zeros((1, width), dtype=torch.float32)
            self.action_std = torch.ones((1, width), dtype=torch.float32)

        if len(all_proprios) > 0:
            all_proprios = torch.from_numpy(np.concatenate(all_proprios, axis=0)).float()  # (N*T, proprio_dim)
            self.proprio_mean = all_proprios.mean(0).unsqueeze(0)  # (1, proprio_dim)
            self.proprio_std = all_proprios.std(0).unsqueeze(0)    # (1, proprio_dim)
        else:
            # Fallback: infer proprio dim from an example in proprio_data or default to 1
            example_shape = None
            for v in self.proprio_data.values():
                example = np.asarray(v)
                if example.size > 0:
                    example_shape = example.shape
                    break
            if example_shape is not None:
                pwidth = int(example_shape[1])
            else:
                pwidth = 1
            logging.warning("No proprio clips found for normalization; falling back to zeros/ones of width %s", pwidth)
            self.proprio_mean = torch.zeros((1, pwidth), dtype=torch.float32)
            self.proprio_std = torch.ones((1, pwidth), dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_id, start_idx = self.samples[idx]
        
        # clip_len = n_steps * frameskip raw frames
        clip_len = self.n_steps * self.frameskip
        
        # Get frame indices with frameskip for slots (matching VideoDataset: data[::frameskip])
        frame_indices = [start_idx + i * self.frameskip for i in range(self.n_steps)]
        
        # Extract slot embeddings: (n_steps, num_slots, slot_dim)
        slots = self.slot_data[video_id]
        pixels_embed = torch.from_numpy(slots[frame_indices]).float()
        
        # Extract and reshape actions (matching VideoDataset behavior)
        # VideoDataset gets clip_len consecutive raw actions, then reshapes to (n_steps, -1)
        action_raw = self.action_data[video_id]
        action_clip = action_raw[start_idx:start_idx + clip_len]  # (clip_len, action_dim)
        # Reshape to (n_steps, action_dim * frameskip) - matching VideoDataset's reshape
        action = torch.from_numpy(action_clip.reshape(self.n_steps, -1)).float()
        
        # Extract proprio with frameskip (matching VideoDataset: data[::frameskip])
        proprio_raw = self.proprio_data[video_id]
        proprio = torch.from_numpy(proprio_raw[frame_indices]).float()
        
        # Normalize action and proprio (matching WrapTorchTransform behavior)
        # Note: No nan_to_num here - that's done in forward pass like train_causalwm.py
        action = (action - self.action_mean) / self.action_std
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        
        sample = {
            "pixels_embed": pixels_embed,  # (T, S, D)
            "action": action,              # (T, action_dim * frameskip)
            "proprio": proprio,            # (T, proprio_dim)
        }
        
        # Optionally include state
        if self.state_data is not None:
            state_raw = self.state_data[video_id]
            state = torch.from_numpy(state_raw[frame_indices]).float()
            sample["state"] = state
        
        return sample

