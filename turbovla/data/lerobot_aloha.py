"""LeRobot -> TurboVLA data adapter for the ALOHA dataset
(thanh0210/aloha_left_arm_pick_carrot_put_cup_easy_task).

Map-style replacement for turbovla.data.libero_rlds.LiberoRLDSDataset that
reads LeRobot v2.1 parquet + mp4 (AV1) episodes instead of RLDS/TFDS, and
emits the exact same batch contract (samples["dinov3"], instructions, states,
action_chunks, action_chunk_masks) via vla_collate_fn so turbovla.models.turbovla
does not need to change.

VERIFIED FACTS FROM TIP-001 (data-driven, not assumptions -- locked into this
module's normalization logic):
  - action = ABSOLUTE joint target, not a delta/residual. corr(action[t],
    state[t+1]) = 1.0000 and mean|action[t] - state[t+1]| = 0.000000 on real
    episodes. Action is therefore normalized exactly like state (same min/max
    mapping to [-1, 1] here), never as action[t] - state[t].
  - gripper (dim index 6) is CONTINUOUS (~107 distinct rounded values, range
    ~1.158-1.632 rad -- a joint angle, not a 0/1 command). It is normalized
    like every other action dimension and is NOT binarized.

Route decision (D4): implemented as a self-contained pyav-based loader
(fallback route) rather than reusing
third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot.LeRobotSingleDataset.
See the Completion Report for TIP-002 for the coupling reasons (missing
meta/modality.json, meta/stats_gr00t.json; pydantic schema + its own
ComposedModalityTransform stack; decord/opencv extra deps) -- integrating it
would fight the exact normalize/contract requirements above rather than
serve them.
"""

from __future__ import annotations

import json
import os
import random

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

STATE_DIM = 7
ACTION_DIM = 7

CAMERA_HIGH = "observation.images.color.high"
CAMERA_WRIST = "observation.images.color.wrist_left"
CAMERA_KEYS = (CAMERA_HIGH, CAMERA_WRIST)


def vla_collate_fn(batch):
    """Verbatim copy of turbovla.data.libero_rlds.vla_collate_fn.

    NOT a reimplementation with different behavior -- byte-for-byte the same
    logic. Copied instead of imported because turbovla/data/libero_rlds.py
    does `import tensorflow as tf` / `import tensorflow_datasets as tfds` at
    module scope, and TensorFlow is intentionally not installed in the
    turbovla-aloha env (TIP-001/TIP-002 constraints both forbid installing
    it). Importing the name from that module would therefore raise
    ModuleNotFoundError even though this function itself has no TensorFlow
    dependency. turbovla/data/libero_rlds.py itself is NOT modified.
    """
    if len(batch) == 0:
        raise ValueError("empty batch cannot be collated")

    dino_img1_list = []
    dino_img2_list = []
    instructions = []
    action_chunks = []
    action_chunk_masks = []
    states = []

    for images, instruction, state, action_chunk, action_chunk_mask in batch:
        if not isinstance(images, (list, tuple)) or len(images) != 2:
            raise ValueError("Each sample must contain two camera views: (img1, img2)")

        img1, img2 = images
        if "dinov3" not in img1 or "dinov3" not in img2:
            raise ValueError("Each view must contain preprocessed tensor for key 'dinov3'")

        dino_img1_list.append(img1["dinov3"])
        dino_img2_list.append(img2["dinov3"])
        instructions.append(instruction)
        action_chunks.append(action_chunk)
        action_chunk_masks.append(action_chunk_mask)
        states.append(state)

    dino_img1 = torch.stack(dino_img1_list, dim=0)
    dino_img2 = torch.stack(dino_img2_list, dim=0)

    samples = {"dinov3": torch.stack([dino_img1, dino_img2], dim=1)}

    action_chunks = torch.stack(action_chunks, dim=0)
    action_chunk_masks = torch.stack(action_chunk_masks, dim=0)
    states = torch.stack(states, dim=0)

    return samples, instructions, states, action_chunks, action_chunk_masks


def split_by_episode(all_eps: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Splits episode indices into (train_eps, val_eps), split by whole
    episode (never by frame) so no trajectory leaks across the split."""
    episodes = list(all_eps)
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    n_val = max(0, min(n_val, len(shuffled)))
    val_eps = sorted(shuffled[:n_val])
    train_eps = sorted(shuffled[n_val:])
    return train_eps, val_eps


class AlohaLeRobotDataset(Dataset):
    """Map-style Dataset over a LeRobot v2.1 ALOHA episode subset.

    action = absolute joint target (verified TIP-001: corr=1.0 vs state[t+1]);
    gripper dim 6 continuous, normalized like other dims, NOT binarized.

    __getitem__ returns a single (unbatched) sample as the tuple
    ((img1, img2), instruction, state, action_chunk, action_chunk_mask), which
    matches what vla_collate_fn (above) expects to stack into the batch
    contract:
        samples            = {"dinov3": Tensor[B, 2, 3, image_size, image_size]}
        instructions       = list[str] (len B)
        states             = Tensor[B, 7]
        action_chunks      = Tensor[B, chunk_size, 7]
        action_chunk_masks = Tensor[B, chunk_size]
    """

    def __init__(
        self,
        data_root: str,
        episodes: list[int],
        dinov3_path: str,
        stats_path: str,
        stats_key: str = "aloha",
        chunk_size: int = 12,
        image_size: int = 256,
    ) -> None:
        self.data_root = data_root
        self.episodes = list(episodes)
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)

        with open(stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)[stats_key]
        self.state_mean = torch.tensor(stats["state"]["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["state"]["std"], dtype=torch.float32)
        self.action_min = torch.tensor(stats["action"]["min"], dtype=torch.float32)
        self.action_max = torch.tensor(stats["action"]["max"], dtype=torch.float32)
        assert self.state_mean.numel() == STATE_DIM
        assert self.action_min.numel() == ACTION_DIM

        self.dino_processor = AutoImageProcessor.from_pretrained(dinov3_path, local_files_only=True)
        if hasattr(self.dino_processor, "do_resize"):
            self.dino_processor.do_resize = False
        if hasattr(self.dino_processor, "do_center_crop"):
            self.dino_processor.do_center_crop = False

        self.instruction = self._load_instruction()

        self._episode_data_cache: dict[int, dict] = {}
        self._episode_video_cache: dict[tuple[int, str], np.ndarray] = {}

        self._index: list[tuple[int, int]] = []
        for episode_id in self.episodes:
            episode = self._get_episode_data(episode_id)
            self._index.extend((episode_id, t) for t in range(episode["length"]))

    def _load_instruction(self) -> str:
        tasks_path = os.path.join(self.data_root, "meta", "tasks.jsonl")
        with open(tasks_path, "r", encoding="utf-8") as handle:
            first_line = handle.readline()
        return str(json.loads(first_line)["task"])

    def _episode_parquet_path(self, episode_id: int) -> str:
        return os.path.join(self.data_root, "data", "chunk-000", f"episode_{episode_id:06d}.parquet")

    def _episode_video_path(self, episode_id: int, camera_key: str) -> str:
        return os.path.join(
            self.data_root, "videos", "chunk-000", camera_key, f"episode_{episode_id:06d}.mp4"
        )

    def _get_episode_data(self, episode_id: int) -> dict:
        """Reads+caches (state, action) arrays for an episode's parquet file
        once, so repeated __getitem__ calls never re-read the parquet."""
        cached = self._episode_data_cache.get(episode_id)
        if cached is None:
            df = pd.read_parquet(self._episode_parquet_path(episode_id))
            state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            action = np.stack(df["action"].to_numpy()).astype(np.float32)
            cached = {"state": state, "action": action, "length": len(df)}
            self._episode_data_cache[episode_id] = cached
        return cached

    @staticmethod
    def _decode_all_frames(video_path: str) -> np.ndarray:
        container = av.open(video_path)
        try:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        finally:
            container.close()
        return np.stack(frames, axis=0)

    def _get_episode_video(self, episode_id: int, camera_key: str) -> np.ndarray:
        """Decodes+caches an episode's full video (all frames, one sequential
        pyav decode pass) once, so __getitem__ never reopens the container
        per frame. Frame i of the mp4 corresponds 1:1 to row i of the
        parquet (verified for episodes 0-7 in both camera views before
        writing this loader -- same 15fps, same episode length)."""
        key = (episode_id, camera_key)
        cached = self._episode_video_cache.get(key)
        if cached is None:
            cached = self._decode_all_frames(self._episode_video_path(episode_id, camera_key))
            episode_length = self._get_episode_data(episode_id)["length"]
            if cached.shape[0] != episode_length:
                raise ValueError(
                    f"episode {episode_id} camera {camera_key!r}: mp4 has {cached.shape[0]} frames "
                    f"but parquet has {episode_length} rows -- frame/row sync assumption violated, "
                    "stopping rather than guessing an alignment."
                )
            self._episode_video_cache[key] = cached
        return cached

    def _process_image(self, frame_hwc_uint8: np.ndarray) -> dict:
        """Resize-distort 480x640 -> image_size x image_size (D8: keeps all
        content, consistent train/eval distortion, no center-crop), then run
        through the DINOv3 AutoImageProcessor with do_resize/do_center_crop
        disabled so it only normalizes (mirrors libero_rlds._process_image_pair)."""
        img = Image.fromarray(frame_hwc_uint8).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        pixel_values = self.dino_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        if tuple(pixel_values.shape[-2:]) != (self.image_size, self.image_size):
            raise RuntimeError(
                f"DINOv3 preprocessor changed image size to {tuple(pixel_values.shape[-2:])}; "
                "do_resize/do_center_crop should be disabled."
            )
        return {"dinov3": pixel_values}

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / (self.state_std + 1e-6)

    def _normalize_action_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Maps ALL 7 action dims (including the continuous gripper, dim 6)
        to [-1, 1] via min/max, matching the state normalization approach.
        Action is an absolute joint target (TIP-001), so this uses the exact
        same per-frame semantics as state -- no delta/residual, no binarize."""
        normed = 2.0 * (action_chunk - self.action_min) / (self.action_max - self.action_min + 1e-6) - 1.0
        return normed.clamp(-1.0, 1.0)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int):
        episode_id, t = self._index[index]
        episode = self._get_episode_data(episode_id)
        length = episode["length"]

        high_frames = self._get_episode_video(episode_id, CAMERA_HIGH)
        wrist_frames = self._get_episode_video(episode_id, CAMERA_WRIST)

        img1 = self._process_image(high_frames[t])
        img2 = self._process_image(wrist_frames[t])

        state = torch.from_numpy(episode["state"][t].copy())
        state = self._normalize_state(state)

        end = min(t + self.chunk_size, length)
        valid_len = end - t
        action_slice = episode["action"][t:end]
        if valid_len < self.chunk_size:
            pad_len = self.chunk_size - valid_len
            pad = np.repeat(action_slice[-1:], pad_len, axis=0)
            action_chunk_np = np.concatenate([action_slice, pad], axis=0)
        else:
            action_chunk_np = action_slice

        action_chunk = torch.from_numpy(action_chunk_np.copy())
        action_chunk = self._normalize_action_chunk(action_chunk)

        action_chunk_mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        action_chunk_mask[:valid_len] = 1.0

        return (img1, img2), self.instruction, state, action_chunk, action_chunk_mask
