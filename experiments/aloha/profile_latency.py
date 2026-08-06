"""Latency profiler for TurboVLA built with the ALOHA config (D2, same as
TIP-003): per-component (text_encoder / vision / vision_language_interaction /
action_head / end-to-end) and, optionally, per-layer (DINOv3 blocks, BERT
layers, fusion+text interaction layers, action decoder layers). No training,
no simulator -- pure forward-pass timing on GPU, random weights are fine
(shapes don't depend on trained values).

Module paths hooked (confirmed against the actual code, not assumed):
    model.text_encoder                                   (whole component)
    model.vision_encoder + model.vision_projection        (merged into "vision")
    model.vision_language_interaction                    (whole component)
    model.action_head                                     (whole component)
    model.vision_encoder.backbone.layer[i]                (DINOv3 ViT blocks, i=0..11)
    model.text_encoder.bert.encoder.layer[i]              (BERT layers, i=0..11)
    model.vision_language_interaction.fusion_layers[i]    (i=0..5)
    model.vision_language_interaction.text_layers[i]      (i=0..5)
    model.action_head.decoder.decoder.layers[j]           (ACT transformer decoder layers)

Usage:
    python -m experiments.aloha.profile_latency --per-layer --precision bf16
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from types import SimpleNamespace

import numpy as np
import torch

from turbovla.models.turbovla import build_turbovla
from experiments.aloha.verify_init import load_with_report

ACTION_DIM = 7
STATE_DIM = 7
CHUNK_SIZE = 12
NUM_VIEWS = 2
IMAGE_SIZE = 256
DEFAULT_INSTRUCTION = "Grasp the carrot from the plate, hold it, place it into the cup."

# Reference numbers from the TurboVLA paper on RTX 4090 -- comparison only, NOT a target.
PAPER_REFERENCE_RTX4090 = {"e2e_ms": 31.0, "vram_gb": 0.9, "hz": 32.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboVLA ALOHA latency profiler.")
    parser.add_argument("--init_ckpt", type=str, default=None,
                         help="Optional. If given, loads weights (shape-filtered like TIP-003/verify_init.py). "
                              "If omitted, random init is used -- latency is unaffected since shapes don't change.")
    parser.add_argument("--dinov3_path", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--bert_path", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--per-layer", dest="per_layer", action="store_true")
    parser.add_argument("--torch-profiler", dest="torch_profiler", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/aloha_latency")
    return parser.parse_args()


def build_aloha_model_args(dinov3_path: str, bert_path: str) -> SimpleNamespace:
    """D2 ALOHA config -- identical to TIP-003's build_model_args (no optimizer
    fields needed here since we never train)."""
    args = SimpleNamespace()
    args.dinov3_path = dinov3_path
    args.bert_path = bert_path
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
    args.action_dim = ACTION_DIM
    args.chunk_size = CHUNK_SIZE
    args.state_dim = STATE_DIM
    args.num_state_tokens = 2
    args.local_files_only = True
    args.freeze_vision_encoder = False
    args.freeze_text_encoder = True
    args.dinov3_precision = "bf16_autocast"
    args.num_views = NUM_VIEWS
    args.image_size = IMAGE_SIZE
    args.position_embedding = "view"
    args.encode_views_separately = True
    args.padding_strategy = "key_padding_mask"
    return args


class HookTimer:
    """Attaches forward pre/post hooks to a set of (module, label) pairs and
    records per-call elapsed ms (cuda Event based, synchronized on each post
    hook -- see module docstring: this is the documented per-layer sync
    caveat, used deliberately here for BOTH component and layer timing)."""

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}
        self._pending: dict[int, torch.cuda.Event] = {}
        self._handles = []

    def attach(self, module: torch.nn.Module, label: str) -> None:
        self.records.setdefault(label, [])

        def pre_hook(mod, inp):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._pending[id(mod)] = start

        def post_hook(mod, inp, out):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            start = self._pending.pop(id(mod))
            self.records[label].append(start.elapsed_time(end))

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def detach_all(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def make_fake_batch(batch_size: int, device: torch.device):
    samples = {"dinov3": torch.randn(batch_size, NUM_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)}
    state = torch.randn(batch_size, STATE_DIM, device=device)
    instructions = [DEFAULT_INSTRUCTION] * batch_size
    return samples, instructions, state


def summarize(values_ms: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(values_ms),
        "median_ms": statistics.median(values_ms),
        "std_ms": statistics.pstdev(values_ms) if len(values_ms) > 1 else 0.0,
        "n": len(values_ms),
    }


def run_warmup(model, samples, instructions, state, autocast_enabled, n_warmup):
    with torch.inference_mode():
        for _ in range(n_warmup):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                model(instructions, samples, state)
    torch.cuda.synchronize()


def profile_components(model, samples, instructions, state, autocast_enabled, warmup, iters, device):
    # Warmup BEFORE attaching hooks, so hook records only ever contain the
    # `iters` timed calls -- not the warmup calls (TIP: "không báo số của vòng warmup").
    run_warmup(model, samples, instructions, state, autocast_enabled, warmup)

    timer = HookTimer()
    timer.attach(model.text_encoder, "text_encoder")
    timer.attach(model.vision_encoder, "vision_encoder_sub")
    timer.attach(model.vision_projection, "vision_projection_sub")
    timer.attach(model.vision_language_interaction, "vision_language_interaction")
    timer.attach(model.action_head, "action_head")

    torch.cuda.reset_peak_memory_stats(device)
    e2e_ms = []
    with torch.inference_mode():
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                model(instructions, samples, state)
            end.record()
            torch.cuda.synchronize()
            e2e_ms.append(start.elapsed_time(end))
    peak_vram_bytes = torch.cuda.max_memory_allocated(device)

    timer.detach_all()

    vision_ms = [a + b for a, b in zip(timer.records["vision_encoder_sub"], timer.records["vision_projection_sub"])]

    component_records = {
        "text_encoder": timer.records["text_encoder"],
        "vision": vision_ms,
        "vision_language_interaction": timer.records["vision_language_interaction"],
        "action_head": timer.records["action_head"],
    }
    return component_records, e2e_ms, peak_vram_bytes


def collect_layer_modules(model) -> list[tuple[str, torch.nn.Module]]:
    layers = []
    missing_notes = []

    try:
        dinov3_layers = model.vision_encoder.backbone.layer
        for i, layer in enumerate(dinov3_layers):
            layers.append((f"dinov3.layer.{i}", layer))
    except AttributeError as exc:
        missing_notes.append(f"model.vision_encoder.backbone.layer not found ({exc})")

    try:
        bert_layers = model.text_encoder.bert.encoder.layer
        for i, layer in enumerate(bert_layers):
            layers.append((f"bert.layer.{i}", layer))
    except AttributeError as exc:
        missing_notes.append(f"model.text_encoder.bert.encoder.layer not found ({exc})")

    try:
        for i, layer in enumerate(model.vision_language_interaction.fusion_layers):
            layers.append((f"interaction.fusion.{i}", layer))
        for i, layer in enumerate(model.vision_language_interaction.text_layers):
            layers.append((f"interaction.text.{i}", layer))
    except AttributeError as exc:
        missing_notes.append(f"model.vision_language_interaction.{{fusion_layers,text_layers}} not found ({exc})")

    try:
        for j, layer in enumerate(model.action_head.decoder.decoder.layers):
            layers.append((f"action_decoder.layer.{j}", layer))
    except AttributeError as exc:
        missing_notes.append(f"model.action_head.decoder.decoder.layers not found ({exc})")

    return layers, missing_notes


def profile_layers(model, samples, instructions, state, autocast_enabled, warmup, iters):
    # Same ordering fix as profile_components: warmup runs with no hooks
    # attached, so per-layer records only ever contain the `iters` timed calls.
    run_warmup(model, samples, instructions, state, autocast_enabled, warmup)

    layer_modules, missing_notes = collect_layer_modules(model)
    timer = HookTimer()
    for label, module in layer_modules:
        timer.attach(module, label)

    with torch.inference_mode():
        for _ in range(iters):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                model(instructions, samples, state)

    timer.detach_all()
    return timer.records, missing_notes


def print_component_table(component_summaries: dict, e2e_summary: dict) -> None:
    e2e_mean = e2e_summary["mean_ms"]
    print(f"\n{'component':<28}{'mean_ms':>10}{'median_ms':>12}{'std_ms':>10}{'% of e2e':>12}")
    total_mean = 0.0
    total_median = 0.0
    for label, summary in component_summaries.items():
        pct = 100.0 * summary["mean_ms"] / e2e_mean if e2e_mean > 0 else float("nan")
        total_mean += summary["mean_ms"]
        total_median += summary["median_ms"]
        print(f"{label:<28}{summary['mean_ms']:>10.3f}{summary['median_ms']:>12.3f}"
              f"{summary['std_ms']:>10.3f}{pct:>11.1f}%")
    print(f"{'sum of 4 components':<28}{total_mean:>10.3f}{total_median:>12.3f}")
    print(f"{'end_to_end (measured)':<28}{e2e_summary['mean_ms']:>10.3f}{e2e_summary['median_ms']:>12.3f}"
          f"{e2e_summary['std_ms']:>10.3f}{100.0:>11.1f}%")

    mean_diff_pct = 100.0 * (total_mean - e2e_mean) / e2e_mean if e2e_mean > 0 else float("nan")
    e2e_median = e2e_summary["median_ms"]
    median_diff_pct = 100.0 * (total_median - e2e_median) / e2e_median if e2e_median > 0 else float("nan")
    print(f"(sum-of-4 vs end-to-end, MEAN:   {total_mean:.3f}ms vs {e2e_mean:.3f}ms, diff={mean_diff_pct:+.1f}%)")
    print(f"(sum-of-4 vs end-to-end, MEDIAN: {total_median:.3f}ms vs {e2e_median:.3f}ms, diff={median_diff_pct:+.1f}% "
          f"-- median is the more robust comparison when any component has high variance/outliers)")


def print_layer_table(layer_summaries: dict, top_n: int = 5) -> None:
    if not layer_summaries:
        print("\n(no per-layer data -- --per-layer not set or no layers were hookable)")
        return
    ranked = sorted(layer_summaries.items(), key=lambda kv: kv[1]["mean_ms"], reverse=True)
    print(f"\nper-layer (top {top_n} slowest, CUDA-synced per hook -- relative comparison only, "
          f"do NOT sum to match end-to-end):")
    print(f"{'layer':<28}{'mean_ms':>10}{'median_ms':>12}{'std_ms':>10}")
    for label, summary in ranked[:top_n]:
        print(f"{label:<28}{summary['mean_ms']:>10.3f}{summary['median_ms']:>12.3f}{summary['std_ms']:>10.3f}")

    prefixes = ["dinov3", "bert", "interaction.fusion", "interaction.text", "action_decoder"]
    print("\nper-layer group totals (sum of mean_ms within group, for a rough block-share comparison):")
    for prefix in prefixes:
        group = {k: v for k, v in layer_summaries.items() if k.startswith(prefix)}
        if group:
            group_total = sum(v["mean_ms"] for v in group.values())
            print(f"  {prefix:<20} n_layers={len(group):<3} sum_mean_ms={group_total:.3f}")


def write_csv(output_path: str, component_summaries: dict, e2e_summary: dict, layer_summaries: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "name", "mean_ms", "median_ms", "std_ms", "n", "pct_of_e2e"])
        e2e_mean = e2e_summary["mean_ms"]
        for label, summary in component_summaries.items():
            pct = 100.0 * summary["mean_ms"] / e2e_mean if e2e_mean > 0 else float("nan")
            writer.writerow(["component", label, f"{summary['mean_ms']:.4f}", f"{summary['median_ms']:.4f}",
                              f"{summary['std_ms']:.4f}", summary["n"], f"{pct:.2f}"])
        writer.writerow(["component", "end_to_end", f"{e2e_summary['mean_ms']:.4f}",
                          f"{e2e_summary['median_ms']:.4f}", f"{e2e_summary['std_ms']:.4f}",
                          e2e_summary["n"], "100.00"])
        for label, summary in layer_summaries.items():
            writer.writerow(["layer", label, f"{summary['mean_ms']:.4f}", f"{summary['median_ms']:.4f}",
                              f"{summary['std_ms']:.4f}", summary["n"], ""])
    print(f"\nwrote: {output_path}")


def run_torch_profiler(model, samples, instructions, state, autocast_enabled, output_dir, n_iters=20):
    trace_path = os.path.join(output_dir, "trace.json")
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(n_iters):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    model(instructions, samples, state)
                prof.step()
    prof.export_chrome_trace(trace_path)
    print(f"wrote: {trace_path}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    autocast_enabled = args.precision == "bf16"

    os.makedirs(args.output_dir, exist_ok=True)

    model_args = build_aloha_model_args(args.dinov3_path, args.bert_path)
    model = build_turbovla(model_args)

    if args.init_ckpt:
        print(f"loading init checkpoint: {args.init_ckpt}")
        report = load_with_report(model, args.init_ckpt)
        print(f"  loaded={report['num_loaded']}/{report['num_target_keys']}, "
              f"missing={len(report['missing_keys'])}, unexpected={len(report['unexpected_keys'])}")
    else:
        print("no --init_ckpt given: using random initialization (shapes are unaffected, latency is valid).")

    model.to(device)
    model.eval()

    print(f"GPU: {gpu_name}")
    print(f"precision={args.precision} (autocast bf16 enabled={autocast_enabled})")
    print(f"batch={args.batch}, warmup={args.warmup}, iters={args.iters}")

    samples, instructions, state = make_fake_batch(args.batch, device)

    component_records, e2e_ms, peak_vram_bytes = profile_components(
        model, samples, instructions, state, autocast_enabled, args.warmup, args.iters, device,
    )
    component_summaries = {label: summarize(values) for label, values in component_records.items()}
    e2e_summary = summarize(e2e_ms)

    print_component_table(component_summaries, e2e_summary)

    peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    hz = 1000.0 / e2e_summary["mean_ms"] if e2e_summary["mean_ms"] > 0 else float("nan")
    print(f"\npeak VRAM allocated: {peak_vram_gb:.3f} GB")
    print(f"throughput: {hz:.2f} Hz (1000 / mean e2e ms)")

    layer_summaries = {}
    if args.per_layer:
        layer_records, missing_notes = profile_layers(
            model, samples, instructions, state, autocast_enabled, args.warmup, args.iters,
        )
        if missing_notes:
            print("\nWARNING -- some per-layer groups could not be hooked (reporting nothing for these, not "
                  "guessing numbers):")
            for note in missing_notes:
                print(f"  - {note}")
        layer_summaries = {label: summarize(values) for label, values in layer_records.items()}
        print_layer_table(layer_summaries)

    print(f"\nreference (paper, RTX 4090, NOT a target): e2e~{PAPER_REFERENCE_RTX4090['e2e_ms']}ms "
          f"VRAM~{PAPER_REFERENCE_RTX4090['vram_gb']}GB {PAPER_REFERENCE_RTX4090['hz']}Hz")
    print(f"this run ({gpu_name}, {args.precision}): e2e={e2e_summary['mean_ms']:.3f}ms "
          f"VRAM={peak_vram_gb:.3f}GB {hz:.2f}Hz")

    csv_path = os.path.join(args.output_dir, f"latency_{args.precision}.csv")
    write_csv(csv_path, component_summaries, e2e_summary, layer_summaries)

    if args.torch_profiler:
        run_torch_profiler(model, samples, instructions, state, autocast_enabled, args.output_dir)


if __name__ == "__main__":
    main()
