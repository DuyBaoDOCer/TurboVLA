"""Verbatim copies of masked_l1_loss / build_scheduler / build_param_group_optimizer
(+ their _use_weight_decay helper) from turbovla.training.trainer.

Copied instead of imported because turbovla/training/trainer.py does
`from ..data.libero_rlds import (...)` at module scope, and libero_rlds.py does
`import tensorflow as tf` / `import tensorflow_datasets as tfds` at module scope.
TensorFlow is intentionally not installed in the turbovla-aloha env (forbidden by
TIP-001/002/003 constraints), so importing these three functions directly raises
ModuleNotFoundError: No module named 'tensorflow' even though none of the three
functions themselves touch TensorFlow. turbovla/training/trainer.py itself is NOT
modified.

Verified via: `python -c "from turbovla.training.trainer import masked_l1_loss"`
-> ModuleNotFoundError: No module named 'tensorflow' (raised from
turbovla/data/libero_rlds.py line 6).
"""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW


def masked_l1_loss(pred, target, mask):
    l1 = torch.abs(pred - target)
    mask = mask.unsqueeze(-1).float()
    l1 = l1 * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return l1.sum() / denom


def build_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            warmup_scale = float(step + 1) / float(max(1, warmup_steps))
            return 0.1 + 0.9 * warmup_scale

        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _use_weight_decay(name, param):
    if param.ndim <= 1:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias"):
        return False
    if "norm" in lowered or "layernorm" in lowered:
        return False
    return True


def build_param_group_optimizer(model, args):
    head_lr = args.lr if args.head_lr is None else args.head_lr
    head_wd = args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
    grouped = {
        ("dinov3_decay", args.dinov3_lr, args.dinov3_weight_decay): [],
        ("dinov3_no_decay", args.dinov3_lr, 0.0): [],
        ("head_decay", head_lr, head_wd): [],
        ("head_no_decay", head_lr, 0.0): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_dino = name.startswith("vision_encoder.backbone")
        decay = _use_weight_decay(name, param)
        if is_dino and decay:
            key = ("dinov3_decay", args.dinov3_lr, args.dinov3_weight_decay)
        elif is_dino:
            key = ("dinov3_no_decay", args.dinov3_lr, 0.0)
        elif decay:
            key = ("head_decay", head_lr, head_wd)
        else:
            key = ("head_no_decay", head_lr, 0.0)
        grouped[key].append(param)

    param_groups = []
    summary = []
    for (group_name, lr, weight_decay), params in grouped.items():
        if not params:
            continue
        count = sum(p.numel() for p in params)
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": group_name})
        summary.append({"name": group_name, "lr": lr, "weight_decay": weight_decay, "params": count})
    return AdamW(param_groups), summary
