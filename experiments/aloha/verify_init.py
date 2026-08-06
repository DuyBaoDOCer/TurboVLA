"""Verify that pretrained/TurboVLA/checkpoints/libero/object.pth loads into a
TurboVLA model built with the ALOHA config (state_dim=7) via strict=False
transfer. Does NOT train. Only builds the model, loads the checkpoint, and
prints missing/unexpected/shape-mismatched keys.

Usage:
    python -m experiments.aloha.verify_init
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from turbovla.models.turbovla import build_turbovla

DEFAULT_CKPT = "pretrained/TurboVLA/checkpoints/libero/object.pth"


def build_aloha_args(local_files_only: bool = True) -> SimpleNamespace:
    """Mirrors turbovla.training.trainer.build_model_architecture's DummyArgs,
    but with the ALOHA-specific dims from TIP-001 CONTEXT and dropout/droppath=0."""
    args = SimpleNamespace()
    args.dinov3_path = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    args.bert_path = "google-bert/bert-base-uncased"
    args.hidden_dim = 256
    args.nheads = 8
    args.dim_feedforward = 2048
    args.max_text_len = 256
    args.text_padding_length = None
    args.text_padding_length_by_instruction = {}
    args.vla_feature_enhancer_layers = 6
    args.enhancer_inner_dim = 1024
    args.text_dropout = 0.0
    args.fusion_dropout = 0.0
    args.fusion_droppath = 0.0
    args.vision_dropout = 0.0
    args.act_dropout = 0.0
    args.action_dim = 7
    args.chunk_size = 12
    args.state_dim = 7
    args.num_state_tokens = 2
    args.local_files_only = local_files_only
    args.freeze_vision_encoder = False
    args.freeze_text_encoder = True
    args.dinov3_precision = "bf16_autocast"
    args.num_views = 2
    args.image_size = 256
    args.position_embedding = "view"
    args.encode_views_separately = True
    args.padding_strategy = "key_padding_mask"
    return args


def _extract_state_dict(ckpt_obj):
    """Same key handling as turbovla.training.trainer._extract_state_dict."""
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("unsupported checkpoint format")


def load_with_report(model: torch.nn.Module, ckpt_path: str):
    """Loads strict=False, but first filters out keys whose checkpoint shape
    does not match the model's shape (load_state_dict raises a RuntimeError on
    shape mismatch regardless of strict=). Shape-mismatched keys are reported
    separately and folded into `missing` (since they were not actually loaded)."""
    # weights_only=False is required: torch >= 2.6 flipped this default to True,
    # which refuses checkpoints containing anything but plain tensors -- and
    # object.pth carries non-tensor entries (args/config metadata). Passing it
    # explicitly keeps this working identically on torch 2.5 (local) and on the
    # newer torch a Colab runtime may ship. Only ever used on checkpoints this
    # project downloaded or wrote itself.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    loadable = {}
    shape_mismatch = []
    for key, tensor in source_state.items():
        if key not in target_state:
            continue
        if tuple(target_state[key].shape) != tuple(tensor.shape):
            shape_mismatch.append((key, tuple(tensor.shape), tuple(target_state[key].shape)))
            continue
        loadable[key] = tensor

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    missing = sorted(set(missing) | {k for k, _, _ in shape_mismatch})
    return {
        "missing_keys": missing,
        "unexpected_keys": sorted(unexpected),
        "shape_mismatch_keys": shape_mismatch,
        "num_source_keys": len(source_state),
        "num_target_keys": len(target_state),
        "num_loaded": len(loadable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify object.pth -> ALOHA TurboVLA load path.")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument(
        "--allow_hf_download",
        action="store_true",
        help="If backbones are not yet cached locally, allow live download instead of local_files_only=True.",
    )
    args = parser.parse_args()

    model_args = build_aloha_args(local_files_only=not args.allow_hf_download)
    print("[verify_init] building TurboVLA (ALOHA config: action_dim=7, state_dim=7, chunk_size=12, "
          "num_views=2, image_size=256, hidden_dim=256, nheads=8, "
          "vla_feature_enhancer_layers=6, num_state_tokens=2, dropout/droppath=0)")
    model = build_turbovla(model_args)
    model.eval()

    print(f"[verify_init] loading checkpoint: {args.ckpt}")
    report = load_with_report(model, args.ckpt)

    print(f"[verify_init] source keys: {report['num_source_keys']}  "
          f"target keys: {report['num_target_keys']}  loaded: {report['num_loaded']}")

    print(f"\n[verify_init] missing_keys ({len(report['missing_keys'])}):")
    for key in report["missing_keys"]:
        print(f"  - {key}")

    print(f"\n[verify_init] unexpected_keys ({len(report['unexpected_keys'])}):")
    for key in report["unexpected_keys"]:
        print(f"  - {key}")

    print(f"\n[verify_init] shape_mismatch_keys ({len(report['shape_mismatch_keys'])}):")
    for key, src_shape, tgt_shape in report["shape_mismatch_keys"]:
        print(f"  - {key}: checkpoint {src_shape} vs model {tgt_shape}")

    expected_missing = {"state_projection.net.1.weight"}
    actual_missing = set(report["missing_keys"])
    if actual_missing == expected_missing and not report["unexpected_keys"]:
        print("\n[verify_init] RESULT: matches CONTEXT expectation exactly.")
    else:
        print("\n[verify_init] RESULT: DEVIATES from CONTEXT expectation "
              f"(expected missing={sorted(expected_missing)}, got missing={sorted(actual_missing)}, "
              f"unexpected={report['unexpected_keys']}). See TIP-001 CONSTRAINTS: report, do not "
              "modify turbovla/models/*.")


if __name__ == "__main__":
    main()
