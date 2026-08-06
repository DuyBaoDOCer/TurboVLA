"""Open-loop eval (predicted action chunk vs recorded ground truth), no
simulator -- Isaac-GR00T Step-4 style. This is the overfit round's real
accept/reject gate (D14): the smoke-loss threshold in TIP-003 was relaxed
(>10x loss drop + correct init/shape/VRAM = "pipeline OK"), so *proof the
model actually learned the labels* now comes from comparing predicted action
chunks against recorded ground-truth on the SAME episodes it trained on, not
from the final training loss number.

For each episode, at every anchor time t (stepped by --num_open_loop_steps,
default = chunk_size = 12, i.e. non-overlapping full coverage), the model
predicts one action chunk of fixed length H=chunk_size=12 (the model's output
shape [B,12,7] never changes -- only the anchor stride changes). The
prediction is un-normalized back to raw action units using the SAME
action_min/action_max the checkpoint itself was trained with (loaded from the
checkpoint's "stats" dict -- never re-read from aloha_stats.json, so there is
only ever one copy of the stats in play), and compared to the recorded
action[t:t+H] in raw units. No closed-loop rollout.

Usage:
    python -m experiments.aloha.eval_openloop \
        --checkpoint experiments/aloha/runs/.../checkpoints/finetune_step_1500.pth \
        --episodes 0
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import build_turbovla
from turbovla.data.lerobot_aloha import AlohaLeRobotDataset, vla_collate_fn
from experiments.aloha.compute_stats import parse_episode_list

ACTION_DIM = 7
JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
]
NMSE_GATE_THRESHOLD = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboVLA ALOHA open-loop eval (predicted vs ground truth).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=str, default=None,
                         help="e.g. '0-7' or '0,1'. Defaults to the episodes recorded in the checkpoint's args "
                              "(the episodes it was trained on) so the gate check is on the train set by default.")
    parser.add_argument("--data_root", type=str, default="data/aloha")
    parser.add_argument("--dinov3_path", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--stats_path", type=str, default="experiments/aloha/configs/aloha_stats.json")
    parser.add_argument("--stats_key", type=str, default="aloha")
    parser.add_argument("--num_open_loop_steps", type=int, default=None,
                         help="Anchor stride between predicted chunks. Defaults to the model's chunk_size "
                              "(12) -> non-overlapping full coverage. Lower (e.g. 1) = receding-horizon, "
                              "H stays 12, ~12x slower, overlapping predicted segments in the plot.")
    parser.add_argument("--output_dir", type=str, default="outputs/aloha_eval")
    parser.set_defaults(plot=True)
    parser.add_argument("--no_plot", dest="plot", action="store_false")
    return parser.parse_args()


def un_normalize_action(pred_norm: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    """Inverts AlohaLeRobotDataset._normalize_action_chunk's min/max->[-1,1] mapping."""
    return (pred_norm + 1.0) / 2.0 * (action_max - action_min) + action_min


def load_model_and_stats(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = TurboVLAConfig.from_mapping(ckpt["model_config"])
    model = build_turbovla(config)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint model_state_dict did not load strictly: missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()

    stats = ckpt["stats"]
    action_min = np.array(stats["action_min"], dtype=np.float32)
    action_max = np.array(stats["action_max"], dtype=np.float32)
    return model, action_min, action_max, ckpt, config.action.horizon


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: [N, 7] raw-unit arrays (N = number of (t, offset) comparisons,
    possibly with repeats if num_open_loop_steps < chunk_size)."""
    diff = pred - gt
    mse_per_joint = (diff ** 2).mean(axis=0)
    mae_per_joint = np.abs(diff).mean(axis=0)
    var_per_joint = gt.var(axis=0)
    nmse_per_joint = mse_per_joint / np.clip(var_per_joint, 1e-8, None)
    return {
        "n_points": int(pred.shape[0]),
        "mse_per_joint": mse_per_joint.tolist(),
        "mse_total": float(mse_per_joint.mean()),
        "mae_per_joint": mae_per_joint.tolist(),
        "mae_total": float(mae_per_joint.mean()),
        "var_per_joint": var_per_joint.tolist(),
        "nmse_per_joint": nmse_per_joint.tolist(),
        "nmse_total": float(nmse_per_joint.mean()),
        "joints_under_gate": int((nmse_per_joint < NMSE_GATE_THRESHOLD).sum()),
        "pred_min_per_joint": pred.min(axis=0).tolist(),
        "pred_max_per_joint": pred.max(axis=0).tolist(),
        "gt_min_per_joint": gt.min(axis=0).tolist(),
        "gt_max_per_joint": gt.max(axis=0).tolist(),
    }


def evaluate_episode(model, dataset, index_lookup, episode_id, action_min, action_max, chunk_size,
                      num_open_loop_steps, device):
    episode = dataset._get_episode_data(episode_id)
    length = episode["length"]
    gt_action_raw = episode["action"]  # [length, 7], raw units

    all_pred = []
    all_gt = []
    chunks_for_plot = []  # list of (start_t, pred_raw[valid_len, 7])

    t = 0
    while t < length:
        flat_idx = index_lookup[(episode_id, t)]
        sample = dataset[flat_idx]
        samples, instructions, states, _, _ = vla_collate_fn([sample])
        samples = {k: v.to(device) for k, v in samples.items()}
        states = states.to(device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_norm = model(instructions, samples, states)  # [1, chunk_size, 7]

        pred_norm = pred_norm[0].float().cpu().numpy()
        pred_raw = un_normalize_action(pred_norm, action_min, action_max)  # [chunk_size, 7]

        valid_len = min(chunk_size, length - t)
        pred_valid = pred_raw[:valid_len]
        gt_valid = gt_action_raw[t:t + valid_len]

        all_pred.append(pred_valid)
        all_gt.append(gt_valid)
        chunks_for_plot.append((t, pred_valid))

        t += num_open_loop_steps

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    metrics = compute_metrics(all_pred, all_gt)
    return metrics, chunks_for_plot, gt_action_raw, length, all_pred, all_gt


def plot_episode(episode_id, chunks_for_plot, gt_action_raw, length, output_dir):
    fig, axes = plt.subplots(ACTION_DIM, 1, figsize=(10, 2.2 * ACTION_DIM), sharex=True)
    time_axis = np.arange(length)

    for j in range(ACTION_DIM):
        ax = axes[j]
        ax.plot(time_axis, gt_action_raw[:, j], color="tab:blue", linewidth=1.5, label="ground truth")
        for idx, (start_t, pred_valid) in enumerate(chunks_for_plot):
            seg_t = np.arange(start_t, start_t + pred_valid.shape[0])
            ax.plot(seg_t, pred_valid[:, j], color="tab:orange", linewidth=1.2, linestyle="--",
                     label="predicted" if idx == 0 else None)
        ax.set_ylabel(JOINT_NAMES[j], fontsize=9)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("frame")
    fig.suptitle(f"episode {episode_id}: predicted vs ground truth")
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"openloop_{episode_id}.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this eval (matches training precision/backend).")
    device = torch.device("cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    model, action_min, action_max, ckpt, chunk_size = load_model_and_stats(args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint} (global_step={ckpt.get('global_step')})")
    print(f"action_min={action_min.tolist()}")
    print(f"action_max={action_max.tolist()}")
    print(f"chunk_size (H)={chunk_size}")

    num_open_loop_steps = args.num_open_loop_steps or chunk_size
    print(f"num_open_loop_steps={num_open_loop_steps} "
          f"({'non-overlapping full coverage' if num_open_loop_steps == chunk_size else 'overlapping/receding-horizon'})")

    if args.episodes is not None:
        episodes = parse_episode_list(args.episodes)
        episodes_source = "--episodes (explicit)"
    else:
        train_episodes = parse_episode_list(ckpt["args"]["episodes"])
        episodes = train_episodes
        episodes_source = "checkpoint's training episodes (default)"
    print(f"episodes={episodes} (source: {episodes_source})")

    dataset = AlohaLeRobotDataset(
        data_root=args.data_root,
        episodes=episodes,
        dinov3_path=args.dinov3_path,
        stats_path=args.stats_path,
        stats_key=args.stats_key,
        chunk_size=chunk_size,
        image_size=ckpt["model_config"]["vision"]["image_size"],
    )
    index_lookup = {pair: i for i, pair in enumerate(dataset._index)}

    if not np.allclose(dataset.action_min.numpy(), action_min, atol=1e-5) or \
       not np.allclose(dataset.action_max.numpy(), action_max, atol=1e-5):
        print("WARNING: dataset stats (--stats_path) differ from checkpoint's saved stats; "
              "un-normalization still uses the checkpoint's own stats as required by spec.")

    per_episode_results = {}
    png_paths = []
    global_pred = []
    global_gt = []
    train_episodes_set = set(parse_episode_list(ckpt["args"]["episodes"]))

    for episode_id in episodes:
        metrics, chunks_for_plot, gt_action_raw, length, all_pred, all_gt = evaluate_episode(
            model, dataset, index_lookup, episode_id, action_min, action_max, chunk_size,
            num_open_loop_steps, device,
        )
        per_episode_results[episode_id] = metrics
        global_pred.append(all_pred)
        global_gt.append(all_gt)

        is_train_episode = episode_id in train_episodes_set
        gate_note = "(TRAIN episode -- overfit gate)" if is_train_episode else "(reference / generalization only)"
        print(f"\nepisode {episode_id} {gate_note}: n_points={metrics['n_points']} "
              f"mse_total={metrics['mse_total']:.6f} nmse_total={metrics['nmse_total']:.6f} "
              f"joints_under_gate={metrics['joints_under_gate']}/7")
        print(f"{'joint':<18}{'MSE':>12}{'MAE':>12}{'NMSE':>12}")
        for j in range(ACTION_DIM):
            print(f"{JOINT_NAMES[j]:<18}{metrics['mse_per_joint'][j]:>12.6f}"
                  f"{metrics['mae_per_joint'][j]:>12.6f}{metrics['nmse_per_joint'][j]:>12.6f}")

        episode_json_path = os.path.join(args.output_dir, f"openloop_{episode_id}.json")
        with open(episode_json_path, "w", encoding="utf-8") as handle:
            json.dump({
                "episode": episode_id,
                "is_train_episode": is_train_episode,
                "num_open_loop_steps": num_open_loop_steps,
                "chunk_size": chunk_size,
                **metrics,
            }, handle, indent=2)
        print(f"wrote: {episode_json_path}")

        if args.plot:
            png_path = plot_episode(episode_id, chunks_for_plot, gt_action_raw, length, args.output_dir)
            png_paths.append(png_path)
            print(f"wrote: {png_path}")

    overall_metrics = compute_metrics(np.concatenate(global_pred, axis=0), np.concatenate(global_gt, axis=0))
    train_only_mask = [ep in train_episodes_set for ep in episodes]
    if any(train_only_mask):
        train_pred = np.concatenate([p for p, is_train in zip(global_pred, train_only_mask) if is_train], axis=0)
        train_gt = np.concatenate([g for g, is_train in zip(global_gt, train_only_mask) if is_train], axis=0)
        overall_train_metrics = compute_metrics(train_pred, train_gt)
    else:
        overall_train_metrics = None

    print(f"\noverall (all evaluated episodes): mse_total={overall_metrics['mse_total']:.6f} "
          f"nmse_total={overall_metrics['nmse_total']:.6f} joints_under_gate={overall_metrics['joints_under_gate']}/7")
    if overall_train_metrics is not None:
        gate_pass = overall_train_metrics["nmse_total"] < NMSE_GATE_THRESHOLD
        print(f"overall (TRAIN episodes only, the actual gate): nmse_total={overall_train_metrics['nmse_total']:.6f} "
              f"joints_under_gate={overall_train_metrics['joints_under_gate']}/7 "
              f"-> GATE {'PASS' if gate_pass else 'FAIL'} (threshold nmse_total < {NMSE_GATE_THRESHOLD})")

    summary_path = os.path.join(args.output_dir, "openloop_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "checkpoint": args.checkpoint,
            "global_step": ckpt.get("global_step"),
            "episodes": episodes,
            "episodes_source": episodes_source,
            "train_episodes": sorted(train_episodes_set),
            "num_open_loop_steps": num_open_loop_steps,
            "chunk_size": chunk_size,
            "nmse_gate_threshold": NMSE_GATE_THRESHOLD,
            "overall": overall_metrics,
            "overall_train_episodes_only": overall_train_metrics,
            "per_episode": per_episode_results,
            "png_paths": png_paths,
        }, handle, indent=2)
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
