"""Compute per-dimension normalization stats for the ALOHA LeRobot dataset
subset downloaded under data/aloha (thanh0210/aloha_left_arm_pick_carrot_put_cup_easy_task).

Reads observation.state (7-D) and action (7-D) from the requested episode
parquet files, reports mean/std for state and min/max/mean/std for action,
checks the gripper dimension (index 6) range, and estimates whether `action`
represents an absolute joint target or a delta from the current state.

Usage:
    python -m experiments.aloha.compute_stats \
        --data_root data/aloha --episodes 0-7 \
        --out experiments/aloha/configs/aloha_stats.json --stats_key aloha
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

STATE_DIM = 7
ACTION_DIM = 7
GRIPPER_INDEX = 6


def parse_episode_list(spec: str) -> list[int]:
    episodes: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(chunk))
    return sorted(set(episodes))


def episode_path(data_root: str, episode_index: int) -> str:
    return os.path.join(data_root, "data", "chunk-000", f"episode_{episode_index:06d}.parquet")


def load_episode(data_root: str, episode_index: int) -> pd.DataFrame:
    path = episode_path(data_root, episode_index)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"episode parquet not found: {path}")
    return pd.read_parquet(path)


def stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(df[column].to_numpy()).astype(np.float64)


def correlate_absolute_vs_delta(episode_frames: list[pd.DataFrame]) -> dict:
    """For each episode, compares action[t] against state[t+1] (absolute
    hypothesis) and against state[t+1]-state[t] (delta hypothesis) using
    per-dim Pearson correlation and mean absolute error."""
    abs_corrs, delta_corrs = [], []
    abs_maes, delta_maes = [], []
    for df in episode_frames:
        action = stack_column(df, "action")
        state = stack_column(df, "observation.state")
        if len(action) < 2:
            continue
        action_t = action[:-1]
        state_t = state[:-1]
        state_tp1 = state[1:]
        delta_state = state_tp1 - state_t

        for dim in range(ACTION_DIM):
            if np.std(action_t[:, dim]) < 1e-9 or np.std(state_tp1[:, dim]) < 1e-9:
                continue
            abs_corrs.append(np.corrcoef(action_t[:, dim], state_tp1[:, dim])[0, 1])
        for dim in range(ACTION_DIM):
            if np.std(action_t[:, dim]) < 1e-9 or np.std(delta_state[:, dim]) < 1e-9:
                continue
            delta_corrs.append(np.corrcoef(action_t[:, dim], delta_state[:, dim])[0, 1])

        abs_maes.append(np.mean(np.abs(action_t - state_tp1)))
        delta_maes.append(np.mean(np.abs(action_t - delta_state)))

    return {
        "mean_corr_action_vs_state_tp1": float(np.mean(abs_corrs)) if abs_corrs else float("nan"),
        "mean_corr_action_vs_delta_state": float(np.mean(delta_corrs)) if delta_corrs else float("nan"),
        "mean_abs_error_action_vs_state_tp1": float(np.mean(abs_maes)) if abs_maes else float("nan"),
        "mean_abs_error_action_vs_delta_state": float(np.mean(delta_maes)) if delta_maes else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ALOHA state/action normalization stats.")
    parser.add_argument("--data_root", type=str, default="data/aloha")
    parser.add_argument("--episodes", type=str, default="0-7", help="e.g. '0-7' or '0,1,2,5'")
    parser.add_argument("--out", type=str, default="experiments/aloha/configs/aloha_stats.json")
    parser.add_argument("--stats_key", type=str, default="aloha")
    args = parser.parse_args()

    episode_indices = parse_episode_list(args.episodes)
    print(f"[compute_stats] loading episodes {episode_indices} from {args.data_root}")

    frames = [load_episode(args.data_root, idx) for idx in episode_indices]
    total_frames = sum(len(df) for df in frames)
    print(f"[compute_stats] loaded {len(frames)} episodes, {total_frames} frames total")

    state = np.concatenate([stack_column(df, "observation.state") for df in frames], axis=0)
    action = np.concatenate([stack_column(df, "action") for df in frames], axis=0)
    assert state.shape[1] == STATE_DIM, f"expected state dim {STATE_DIM}, got {state.shape[1]}"
    assert action.shape[1] == ACTION_DIM, f"expected action dim {ACTION_DIM}, got {action.shape[1]}"

    state_mean = state.mean(axis=0)
    state_std = state.std(axis=0)
    action_min = action.min(axis=0)
    action_max = action.max(axis=0)
    action_mean = action.mean(axis=0)
    action_std = action.std(axis=0)

    assert not np.isnan(state_mean).any() and not np.isnan(state_std).any(), "NaN in state stats"
    assert not np.isnan(action_min).any() and not np.isnan(action_max).any(), "NaN in action stats"
    assert not np.isnan(action_mean).any() and not np.isnan(action_std).any(), "NaN in action stats"

    for name, std in [("state", state_std), ("action", action_std)]:
        near_zero = np.where(std < 1e-6)[0]
        if len(near_zero) > 0:
            print(f"[compute_stats] WARNING: {name} dims with std~=0: {near_zero.tolist()} "
                  f"(std={std[near_zero].tolist()})")

    print(f"\n[compute_stats] gripper (index {GRIPPER_INDEX}) ranges:")
    print(f"  action[:,{GRIPPER_INDEX}]: min={action[:, GRIPPER_INDEX].min():.6f} "
          f"max={action[:, GRIPPER_INDEX].max():.6f} "
          f"unique_values~{len(np.unique(np.round(action[:, GRIPPER_INDEX], 3)))}")
    print(f"  state[:,{GRIPPER_INDEX}]:  min={state[:, GRIPPER_INDEX].min():.6f} "
          f"max={state[:, GRIPPER_INDEX].max():.6f} "
          f"unique_values~{len(np.unique(np.round(state[:, GRIPPER_INDEX], 3)))}")
    gripper_unique = len(np.unique(np.round(action[:, GRIPPER_INDEX], 3)))
    gripper_conclusion = "binary/discrete" if gripper_unique <= 5 else "continuous"
    print(f"  -> gripper dim looks {gripper_conclusion} (heuristic: <=5 distinct rounded values = binary)")

    corr_report = correlate_absolute_vs_delta(frames)
    print("\n[compute_stats] absolute-vs-delta action check "
          f"(episodes {episode_indices}):")
    print(f"  mean corr(action[t], state[t+1])         = {corr_report['mean_corr_action_vs_state_tp1']:.4f}")
    print(f"  mean corr(action[t], state[t+1]-state[t]) = {corr_report['mean_corr_action_vs_delta_state']:.4f}")
    print(f"  mean |action[t] - state[t+1]|              = {corr_report['mean_abs_error_action_vs_state_tp1']:.6f}")
    print(f"  mean |action[t] - (state[t+1]-state[t])|   = {corr_report['mean_abs_error_action_vs_delta_state']:.6f}")
    if corr_report["mean_abs_error_action_vs_state_tp1"] < corr_report["mean_abs_error_action_vs_delta_state"]:
        action_conclusion = "absolute joint target (action[t] ~= state[t+1])"
    else:
        action_conclusion = "delta joint command (action[t] ~= state[t+1] - state[t])"
    print(f"  -> action looks like: {action_conclusion}")

    stats = {
        args.stats_key: {
            "state": {
                "mean": state_mean.tolist(),
                "std": state_std.tolist(),
            },
            "action": {
                "min": action_min.tolist(),
                "max": action_max.tolist(),
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
            },
        }
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"\n[compute_stats] wrote {args.out}")


if __name__ == "__main__":
    main()
