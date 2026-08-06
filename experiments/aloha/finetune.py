"""Single-GPU overfit-smoke finetune trainer for TurboVLA on the ALOHA LeRobot
subset (thanh0210/aloha_left_arm_pick_carrot_put_cup_easy_task).

Calls turbovla.models.turbovla.build_turbovla directly -- does NOT go through
the turbovla/training/trainer.py CLI. Purpose: prove the pipeline (data ->
model -> loss -> optimizer -> checkpoint) runs end-to-end and that the model
can overfit 8 episodes, as a smoke gate before Colab full training (TIP-007).

Usage:
    python -m experiments.aloha.finetune --episodes 0-7 \
        --init_ckpt pretrained/TurboVLA/checkpoints/libero/object.pth \
        --max_steps 1500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from turbovla.models.turbovla import build_turbovla
from turbovla.data.lerobot_aloha import AlohaLeRobotDataset, vla_collate_fn

from experiments.aloha._trainer_utils import (
    build_param_group_optimizer,
    build_scheduler,
    masked_l1_loss,
)
from experiments.aloha.compute_stats import parse_episode_list
from experiments.aloha.verify_init import load_with_report

STATE_DIM = 7
ACTION_DIM = 7
CHUNK_SIZE_DEFAULT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboVLA ALOHA overfit-smoke finetune trainer.")

    parser.add_argument("--data_root", type=str, default="data/aloha")
    parser.add_argument("--episodes", type=str, default="0-7")
    parser.add_argument("--stats_path", type=str, default="experiments/aloha/configs/aloha_stats.json")
    parser.add_argument("--stats_key", type=str, default="aloha")

    parser.add_argument("--dinov3_path", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--bert_path", type=str, default="google-bert/bert-base-uncased")

    parser.add_argument("--init_ckpt", type=str, default="pretrained/TurboVLA/checkpoints/libero/object.pth")
    parser.add_argument("--no_init", action="store_true", help="Skip checkpoint load; random init (error-isolation path).")

    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--vla_feature_enhancer_layers", type=int, default=6)
    parser.add_argument("--num_state_tokens", type=int, default=2)

    parser.set_defaults(freeze_vision_encoder=True)
    parser.add_argument("--no_freeze_vision_encoder", dest="freeze_vision_encoder", action="store_false")
    parser.set_defaults(freeze_text_encoder=True)
    parser.add_argument("--no_freeze_text_encoder", dest="freeze_text_encoder", action="store_false")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--dinov3_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--head_weight_decay", type=float, default=None)
    parser.add_argument("--dinov3_weight_decay", type=float, default=1e-10)

    parser.add_argument("--precision", type=str, default="bf16_amp", choices=["fp32", "bf16_amp"])
    parser.add_argument("--grad_checkpointing", action="store_true")

    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--stop_loss", type=float, default=5e-3)
    parser.add_argument("--stop_patience", type=int, default=3)

    parser.add_argument("--val_every", type=int, default=0, help="0 disables val (TIP-005 not built yet).")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_dir", type=str, default=None, help="Defaults to experiments/aloha/runs/<timestamp>.")
    parser.add_argument("--checkpoint_prefix", type=str, default="finetune_step")

    parser.add_argument("--wandb_project", type=str, default="turbovla-aloha-smoke")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["offline", "online", "disabled"])

    return parser.parse_args()


def build_model_args(args: argparse.Namespace) -> SimpleNamespace:
    """D2 ALOHA model config, passed to build_turbovla the same way
    build_model_architecture in turbovla/training/trainer.py builds its
    DummyArgs object -- but constructed directly here (no CLI trainer)."""
    model_args = SimpleNamespace()
    model_args.dinov3_path = args.dinov3_path
    model_args.bert_path = args.bert_path
    model_args.hidden_dim = args.hidden_dim
    model_args.nheads = args.nheads
    model_args.dim_feedforward = 2048
    model_args.max_text_len = 256
    model_args.text_padding_length = None  # -> "longest" padding
    model_args.text_padding_length_by_instruction = {}
    model_args.vla_feature_enhancer_layers = args.vla_feature_enhancer_layers
    model_args.enhancer_inner_dim = 1024
    model_args.text_dropout = 0.0
    model_args.fusion_dropout = 0.0
    model_args.fusion_droppath = 0.0
    model_args.vision_dropout = 0.0
    model_args.act_dropout = 0.0
    model_args.action_dim = ACTION_DIM
    model_args.chunk_size = args.chunk_size
    model_args.state_dim = STATE_DIM
    model_args.num_state_tokens = args.num_state_tokens
    model_args.local_files_only = True
    model_args.freeze_vision_encoder = args.freeze_vision_encoder
    model_args.freeze_text_encoder = args.freeze_text_encoder
    model_args.dinov3_precision = "bf16_autocast"
    model_args.num_views = 2
    model_args.image_size = args.image_size
    model_args.position_embedding = "view"
    model_args.encode_views_separately = True
    model_args.padding_strategy = "key_padding_mask"
    return model_args


def move_samples_to_device(samples, device):
    if isinstance(samples, dict):
        return {k: v.to(device, non_blocking=True) for k, v in samples.items()}
    return samples.to(device, non_blocking=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this trainer (RTX 4050 smoke target).")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join("experiments", "aloha", "runs", f"finetune_{timestamp}")
    os.makedirs(args.run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(args.run_dir, "train_log.csv")
    jsonl_path = os.path.join(args.run_dir, "train_log.jsonl")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "avg", "lr"])
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    import wandb  # local import: only needed once WANDB_MODE is set

    wandb.init(project=args.wandb_project, name=os.path.basename(args.run_dir), config=vars(args), mode=args.wandb_mode)

    print(f"run_dir={args.run_dir}")
    print(f"device={device}, precision={args.precision}")

    episodes = parse_episode_list(args.episodes)
    dataset = AlohaLeRobotDataset(
        data_root=args.data_root,
        episodes=episodes,
        dinov3_path=args.dinov3_path,
        stats_path=args.stats_path,
        stats_key=args.stats_key,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
    )
    print(f"dataset: episodes={episodes}, frames={len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=vla_collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_iter = iter(dataloader)

    model_args = build_model_args(args)
    model = build_turbovla(model_args)

    if args.no_init:
        print("--no_init set: skipping checkpoint load, using random initialization.")
    else:
        print(f"loading init checkpoint: {args.init_ckpt}")
        report = load_with_report(model, args.init_ckpt)
        print(f"  source keys: {report['num_source_keys']}  target keys: {report['num_target_keys']}  "
              f"loaded: {report['num_loaded']}")
        print(f"  missing_keys ({len(report['missing_keys'])}): {report['missing_keys']}")
        print(f"  unexpected_keys ({len(report['unexpected_keys'])}): {report['unexpected_keys']}")
        print(f"  shape_mismatch_keys ({len(report['shape_mismatch_keys'])}):")
        for key, src_shape, tgt_shape in report["shape_mismatch_keys"]:
            print(f"    - {key}: checkpoint {src_shape} vs model {tgt_shape}")

        expected_missing = {
            "action_head.state_projection.net.0.weight",
            "action_head.state_projection.net.0.bias",
            "action_head.state_projection.net.1.weight",
        }
        actual_missing = set(report["missing_keys"])
        if actual_missing != expected_missing or report["unexpected_keys"]:
            raise RuntimeError(
                "init checkpoint load deviates from the expected 3-key shape mismatch "
                f"(expected missing={sorted(expected_missing)}, got missing={sorted(actual_missing)}, "
                f"unexpected={report['unexpected_keys']}). Stopping per TIP-003 CONSTRAINTS "
                "(do not modify turbovla/models/*; report instead)."
            )
        print("  init load matches the expected 3-key deviation exactly (669/672 transferred).")

    model.to(device)
    model.train()
    if args.freeze_vision_encoder:
        model.dinov3.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,}")

    optimizer, optimizer_summary = build_param_group_optimizer(model, args)
    print("optimizer param groups:", optimizer_summary)
    scheduler = build_scheduler(optimizer, max_steps=args.max_steps, warmup_steps=args.warmup_steps,
                                 min_lr_ratio=args.min_lr_ratio)

    torch.cuda.reset_peak_memory_stats(device)

    global_step = 0
    loss_window: list[float] = []
    stop_streak = 0
    start_time = time.time()

    while global_step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(args.grad_accum_steps):
            try:
                samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)

            samples = move_samples_to_device(samples, device)
            states = states.to(device, non_blocking=True)
            gt_actions = gt_actions.to(device, non_blocking=True)
            action_chunk_masks = action_chunk_masks.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16_amp"):
                pred_actions = model(instructions, samples, states)
                if pred_actions.shape != gt_actions.shape:
                    raise ValueError(
                        f"pred_actions.shape={tuple(pred_actions.shape)} != "
                        f"gt_actions.shape={tuple(gt_actions.shape)}"
                    )
                loss = masked_l1_loss(pred_actions, gt_actions, action_chunk_masks)
            (loss / args.grad_accum_steps).backward()
            loss_accum += loss.detach().item()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        local_loss = loss_accum / args.grad_accum_steps
        loss_window.append(local_loss)
        if len(loss_window) > args.log_freq:
            loss_window.pop(0)
        avg_window_loss = sum(loss_window) / len(loss_window)

        global_step += 1
        current_lr = optimizer.param_groups[0]["lr"]

        if global_step % args.log_freq == 0 or global_step == 1:
            elapsed = time.time() - start_time
            print(f"step={global_step}/{args.max_steps} loss={local_loss:.5f} avg={avg_window_loss:.5f} "
                  f"lr={current_lr:.2e} elapsed={elapsed:.1f}s")
            csv_writer.writerow([global_step, f"{local_loss:.6f}", f"{avg_window_loss:.6f}", f"{current_lr:.8f}"])
            csv_file.flush()
            jsonl_file.write(json.dumps({
                "step": global_step, "loss": local_loss, "avg": avg_window_loss, "lr": current_lr,
                "elapsed_s": elapsed,
            }) + "\n")
            jsonl_file.flush()
            wandb.log({"loss": local_loss, "avg_loss": avg_window_loss, "lr": current_lr}, step=global_step)

            if avg_window_loss < args.stop_loss:
                stop_streak += 1
            else:
                stop_streak = 0

        if args.val_every > 0 and global_step % args.val_every == 0:
            # Open-loop MSE validation belongs to TIP-005 (not built yet); stubbed
            # off by default (--val_every 0) so this TIP does not block on it.
            print(f"[val] skipped: open-loop eval (TIP-005) not implemented yet (val_every={args.val_every})")

        should_save = global_step % args.save_steps == 0 or global_step == args.max_steps
        should_stop = stop_streak >= args.stop_patience
        if should_save or should_stop:
            save_path = os.path.join(checkpoint_dir, f"{args.checkpoint_prefix}_{global_step}.pth")
            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": local_loss,
                    "avg_loss": avg_window_loss,
                    "args": vars(args),
                    "model_config": model.config.to_dict(),
                    "stats": {
                        "state_mean": dataset.state_mean.tolist(),
                        "state_std": dataset.state_std.tolist(),
                        "action_min": dataset.action_min.tolist(),
                        "action_max": dataset.action_max.tolist(),
                        "stats_key": args.stats_key,
                    },
                },
                save_path,
            )
            print(f"saved: {save_path}")

        if should_stop:
            print(f"early-stop: avg_loss < {args.stop_loss} for {args.stop_patience} consecutive log checkpoints "
                  f"at step {global_step}.")
            break

    peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    print(f"peak VRAM allocated: {peak_vram_bytes / (1024 ** 3):.3f} GB")
    print(f"total steps: {global_step}, total time: {time.time() - start_time:.1f}s")

    csv_file.close()
    jsonl_file.close()
    wandb.finish()


if __name__ == "__main__":
    main()
