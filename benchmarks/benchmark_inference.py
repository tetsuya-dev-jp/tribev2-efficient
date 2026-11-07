"""Benchmark TRIBE v2 inference variants on the same inputs and device."""

import argparse
import gc
import json
import math
import os
import platform
import sys
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from tribev2.demo_utils import TribeModel

UPSTREAM_COMMIT = "af58661791a351a448a489042a28f6c37e1c14b7"
CHECKPOINT_NAME = "best.ckpt"
RELEASED_TIMESTEPS = 200
SYNTHETIC_SEED = 0
RTOL = 1e-4
ATOL = 1e-5
RSS_INTERVAL_MS = 5
TIMING_SCOPE = "forward_only_excludes_H2D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TRIBE v2 inference.")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Directory containing config.yaml and best.ckpt.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=Path, default=None,
                        help="Optional torch.save list of CPU feature dictionaries.")
    parser.add_argument("--sdpa", action="store_true",
                        help="Also benchmark SDPA-only and fused+SDPA candidates.")
    parser.add_argument("--reverse-order", action="store_true",
                        help="Run conditions in reverse order with fresh models.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.steps < 1 or args.warmup < 1:
        parser.error("--steps and --warmup must be at least 1")
    if not args.checkpoint.is_dir():
        parser.error(f"--checkpoint must be an existing directory: {args.checkpoint}")
    if args.batches is not None and not args.batches.is_file():
        parser.error(f"--batches must be an existing file: {args.batches}")
    return args


def synthetic_batches(feature_dims: dict, batch_size: int) -> list[dict]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SYNTHETIC_SEED)
    batch = {}
    for modality, dims in feature_dims.items():
        if dims is None:
            continue
        n_layers, feature_dim = dims
        shape = (batch_size, int(n_layers), int(feature_dim), RELEASED_TIMESTEPS)
        batch[modality] = torch.randn(shape, generator=generator, dtype=torch.float32)
    if not batch:
        raise ValueError("model.feature_dims has no usable feature dimensions")
    batch["subject_id"] = torch.zeros(batch_size, dtype=torch.long)
    return [batch]


def load_cached_batches(path: Path) -> list[dict]:
    batches = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(batches, list) or not batches:
        raise ValueError("--batches must contain a non-empty list of feature batches")

    normalized = []
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise ValueError(f"batch {index} is not a dictionary")
        data = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"batch {index} field {key!r} is not a torch.Tensor")
            if value.device.type != "cpu":
                raise ValueError(f"batch {index} field {key!r} is not on CPU")
            data[key] = value
        if "subject_id" not in data:
            features = [value for value in data.values() if value.ndim > 0]
            if not features:
                raise ValueError(f"batch {index} has no feature tensors")
            data["subject_id"] = torch.zeros(features[0].shape[0], dtype=torch.long)
        else:
            data["subject_id"] = data["subject_id"].to(torch.long)
        normalized.append(data)
    return normalized


def batch_sizes(batches: list[dict]) -> list[int]:
    return [int(batch["subject_id"].shape[0]) for batch in batches]


def to_device(batch: dict, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        data={key: value.to(device) for key, value in batch.items()}
    )


def parameter_stats(model: torch.nn.Module) -> tuple[int, int]:
    count = sum(parameter.numel() for parameter in model.parameters())
    nbytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    return int(count), int(nbytes)


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    xp = TribeModel.from_pretrained(
        checkpoint, checkpoint_name=CHECKPOINT_NAME, device="cpu", optimize=False
    )
    model = xp._model
    if model is None:
        raise RuntimeError("TribeModel.from_pretrained did not create a model")
    model.eval()
    model.to(device)
    return model


def release_cached_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def predict(
    model: torch.nn.Module, batches: list[dict], device: torch.device
) -> list[torch.Tensor]:
    outputs = []
    with torch.inference_mode():
        for batch in batches:
            batch_device = to_device(batch, device)
            output = model(batch_device)
            outputs.append(output.detach().to(device="cpu", dtype=torch.float32))
            del output, batch_device
    return outputs


def read_rss_bytes() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


class RssSampler:
    """Sample process RSS on a background thread while a conversion runs."""

    def __init__(self, interval_ms: int = RSS_INTERVAL_MS):
        self.interval_ms = interval_ms
        self.baseline = read_rss_bytes()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.baseline is not None:
            self._thread = threading.Thread(target=self._sample_until_stopped)
            self._thread.daemon = True
            self._thread.start()
        return self

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000.0):
            self._record(read_rss_bytes())

    def _record(self, value: int | None) -> None:
        if value is not None and (self.peak is None or value > self.peak):
            self.peak = value

    def __exit__(self, *exc_info):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._record(read_rss_bytes())
        return False

    def result(self) -> dict | None:
        if self.baseline is None:
            return None
        peak = self.peak if self.peak is not None else self.baseline
        return {
            "interval_ms": self.interval_ms,
            "baseline_rss_bytes": int(self.baseline),
            "peak_rss_bytes": int(peak),
            "incremental_rss_bytes": int(max(peak - self.baseline, 0)),
        }


def measure_conversion(model: torch.nn.Module, device: torch.device) -> dict:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
    rss = RssSampler()
    start = time.perf_counter()
    with rss:
        optimized = model.optimize_for_inference()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if optimized is not None and optimized is not model:
        raise RuntimeError("optimize_for_inference must modify the model in place")
    cuda_memory = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated()
        cuda_memory = {
            "peak_allocated_bytes": int(peak_allocated),
            "delta_over_start_bytes": int(max(peak_allocated - start_allocated, 0)),
            "scope": "torch_cuda_allocator_from_reset_stats",
        }
    return {"time_ms": elapsed_ms, "rss": rss.result(), "cuda": cuda_memory}


def time_forward(
    model: torch.nn.Module,
    batches: list[dict],
    device: torch.device,
    steps: int,
    warmup: int,
) -> tuple[list[float], dict]:
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize()
    with torch.inference_mode():
        for index in range(warmup):
            batch_device = to_device(batches[index % len(batches)], device)
            output = model(batch_device)
            del output, batch_device
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated_at_start = torch.cuda.memory_allocated()
        samples = []
        for index in range(steps):
            batch_device = to_device(batches[index % len(batches)], device)
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(batch_device)
            if use_cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
            del output, batch_device
        if use_cuda:
            peak_allocated = torch.cuda.max_memory_allocated()

    cuda = None
    if use_cuda:
        cuda = {
            "allocated_bytes_at_start": int(allocated_at_start),
            "peak_allocated_bytes": int(peak_allocated),
            "incremental_allocated_bytes": int(
                max(peak_allocated - allocated_at_start, 0)
            ),
            "scope": "torch_cuda_allocator_during_forward_timing",
            "note": (
                "Allocator bytes; includes input/output allocations made outside "
                "the timer; not physical VRAM and not activation-only memory."
            ),
        }
    return samples, {"cuda": cuda, "cpu": None}


def summarize_timing(samples: list[float], memory: dict) -> dict:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples_ms": [float(value) for value in samples],
        "median_ms": float(np.median(values)),
        "p25_ms": float(np.percentile(values, 25)),
        "p75_ms": float(np.percentile(values, 75)),
        "timing_scope": TIMING_SCOPE,
        "memory": memory,
    }


def compare_predictions(
    reference: list[torch.Tensor], candidate: list[torch.Tensor], label: str
) -> dict:
    result = {
        "finite_elements": 0,
        "nonfinite_reference_elements": 0,
        "nonfinite_candidate_elements": 0,
        "tolerance_failure_elements": 0,
        "max_abs_error": None,
        "relative_l2": None,
        "centered_correlation": None,
        "rtol": RTOL,
        "atol": ATOL,
        "allclose": False,
    }
    if len(reference) != len(candidate):
        result["error"] = (
            f"{label} produced {len(candidate)} outputs, expected {len(reference)}"
        )
        return result

    count = 0
    max_abs = squared_error = reference_squared = 0.0
    sum_reference = sum_candidate = sum_product = candidate_squared = 0.0
    for index, (ref, cand) in enumerate(zip(reference, candidate)):
        if ref.shape != cand.shape:
            result["error"] = (
                f"{label} output {index} has shape {tuple(cand.shape)}, "
                f"expected {tuple(ref.shape)}"
            )
            return result
        ref = ref.detach().reshape(-1).to(torch.float64)
        cand = cand.detach().reshape(-1).to(torch.float64)
        finite_reference = torch.isfinite(ref)
        finite_candidate = torch.isfinite(cand)
        result["nonfinite_reference_elements"] += int((~finite_reference).sum())
        result["nonfinite_candidate_elements"] += int((~finite_candidate).sum())
        valid = finite_reference & finite_candidate
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        ref_valid, cand_valid = ref[valid], cand[valid]
        difference = cand_valid - ref_valid
        count += n_valid
        max_abs = max(max_abs, float(difference.abs().max()))
        squared_error += float(difference.square().sum())
        reference_squared += float(ref_valid.square().sum())
        sum_reference += float(ref_valid.sum())
        sum_candidate += float(cand_valid.sum())
        sum_product += float((ref_valid * cand_valid).sum())
        candidate_squared += float(cand_valid.square().sum())
        close = torch.isclose(cand_valid, ref_valid, rtol=RTOL, atol=ATOL)
        result["tolerance_failure_elements"] += int((~close).sum())

    result["finite_elements"] = count
    if count:
        result["max_abs_error"] = max_abs
    if reference_squared > 0.0:
        result["relative_l2"] = math.sqrt(squared_error / reference_squared)
    if count > 1:
        centered_products = sum_product - sum_reference * sum_candidate / count
        reference_variance = reference_squared - sum_reference**2 / count
        candidate_variance = candidate_squared - sum_candidate**2 / count
        denominator = math.sqrt(
            max(reference_variance, 0.0) * max(candidate_variance, 0.0)
        )
        if denominator > 0.0:
            result["centered_correlation"] = centered_products / denominator
    result["allclose"] = (
        count > 0
        and result["nonfinite_reference_elements"] == 0
        and result["nonfinite_candidate_elements"] == 0
        and result["tolerance_failure_elements"] == 0
    )
    return result


def pooling_only_forward(self, batch, pool_outputs: bool = True):
    x, subject_id = self.encode(batch)
    x = x.transpose(1, 2)
    if pool_outputs:
        x = self.pooler(x)
    if self.config.low_rank_head is not None:
        x = self.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
    return self.predictor(x, subject_id)


def set_sdpa(
    model: torch.nn.Module, enabled: bool
) -> tuple[list | None, list | None, str | None]:
    config = model.config
    if getattr(config, "linear_baseline", False):
        return None, None, "linear baseline has no attention"
    encoder_config = getattr(config, "encoder", None)
    if encoder_config is None:
        return None, None, "model has no transformer encoder"
    if getattr(encoder_config, "causal", False):
        return None, None, "causal attention is not supported"
    if getattr(encoder_config, "residual_attn", False):
        return None, None, "residual attention is not supported"

    from x_transformers.attend import Attend

    attends = [module for module in model.modules() if isinstance(module, Attend)]
    if not attends:
        return None, None, "no x_transformers Attend modules found"
    for module in attends:
        if getattr(module, "causal", False):
            return None, None, "causal attention is not supported"
        if getattr(module, "residual_attn", False):
            return None, None, "residual attention is not supported"
    originals = [bool(getattr(module, "flash", False)) for module in attends]
    for module in attends:
        module.flash = enabled
    return attends, originals, None


def run_condition(
    name: str,
    checkpoint: Path,
    batches: list[dict],
    device: torch.device,
    steps: int,
    warmup: int,
    baseline_predictions: list[torch.Tensor] | None,
) -> dict:
    model = load_model(checkpoint, device)
    original_forward = None
    attends = None
    flash_originals = None
    conversion = None
    try:
        if name == "pooling_only":
            original_forward = model.forward
            model.forward = MethodType(pooling_only_forward, model)
        elif name == "sdpa_only":
            attends, flash_originals, reason = set_sdpa(model, True)
            if reason is not None:
                return {"timing": {"enabled": False, "reason": reason}}
        elif name in ("optimized", "sdpa"):
            conversion = measure_conversion(model, device)
            if name == "sdpa":
                attends, flash_originals, reason = set_sdpa(model, True)
                if reason is not None:
                    return {
                        "timing": {"enabled": False, "reason": reason},
                        "conversion": conversion,
                    }

        parameter_count, parameter_bytes = parameter_stats(model)
        samples, memory = time_forward(model, batches, device, steps, warmup)
        correctness = None
        if baseline_predictions is not None:
            predictions = predict(model, batches, device)
            correctness = compare_predictions(baseline_predictions, predictions, name)
        return {
            "timing": summarize_timing(samples, memory),
            "correctness": correctness,
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "conversion": conversion,
        }
    finally:
        if original_forward is not None:
            model.forward = original_forward
        if attends is not None:
            for module, original in zip(attends, flash_originals):
                module.flash = original
        original_forward = attends = flash_originals = None
        del model
        release_cached_memory(device)


def checkpoint_revision(path: Path) -> str | None:
    name = path.resolve().name
    if len(name) == 40 and all(c in "0123456789abcdef" for c in name):
        return name
    return None


def format_optional(value: float | None) -> str:
    return "null" if value is None else f"{value:.6g}"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        torch.cuda.set_device(device)

    if args.batches is not None:
        batches = load_cached_batches(args.batches)
        input_kind = "cached_feature_batches"
        requested_batch_size = None
    else:
        inspection_model = load_model(args.checkpoint, torch.device("cpu"))
        feature_dims = inspection_model.feature_dims
        del inspection_model
        release_cached_memory(torch.device("cpu"))
        batches = synthetic_batches(feature_dims, args.batch_size)
        input_kind = "synthetic"
        requested_batch_size = args.batch_size
    sizes = batch_sizes(batches)

    baseline_model = load_model(args.checkpoint, device)
    parameter_count, parameter_bytes = parameter_stats(baseline_model)
    parameter_dtype = str(next(baseline_model.parameters()).dtype)
    baseline_predictions = predict(baseline_model, batches, device)
    del baseline_model
    release_cached_memory(device)

    order = ["baseline", "pooling_only"]
    if args.sdpa:
        order.append("sdpa_only")
    order.append("optimized")
    if args.sdpa:
        order.append("sdpa")
    if args.reverse_order:
        order = list(reversed(order))

    results = {}
    for name in order:
        results[name] = run_condition(
            name,
            args.checkpoint,
            batches,
            device,
            args.steps,
            args.warmup,
            None if name == "baseline" else baseline_predictions,
        )

    def entry(name: str, field: str):
        return results.get(name, {}).get(field)

    optimized_conversion = entry("optimized", "conversion")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    result = {
        "benchmark": "tribev2_inference",
        "upstream_commit": UPSTREAM_COMMIT,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "hip": getattr(torch.version, "hip", None),
            "device": str(device),
            "device_name": device_name,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "dtype": parameter_dtype,
        },
        "checkpoint": {
            "name": CHECKPOINT_NAME,
            "revision_basename": checkpoint_revision(args.checkpoint),
        },
        "input": {
            "kind": input_kind,
            "synthetic_seed": SYNTHETIC_SEED if input_kind == "synthetic" else None,
            "n_batches": len(batches),
            "batch_size": requested_batch_size,
            "batch_sizes": sizes,
            "shapes": [
                {key: list(value.shape) for key, value in batch.items()}
                for batch in batches
            ],
        },
        "conditions": {"order": order, "reverse_order": args.reverse_order},
        "model": {
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "optimized_parameter_count": entry("optimized", "parameter_count"),
            "optimized_parameter_bytes": entry("optimized", "parameter_bytes"),
            "conversion_time_ms": (
                None if optimized_conversion is None else optimized_conversion["time_ms"]
            ),
            "conversion_memory": optimized_conversion,
        },
        "timing": {
            "steps": args.steps,
            "warmup": args.warmup,
            "baseline": entry("baseline", "timing"),
            "pooling_only": entry("pooling_only", "timing"),
            "optimized": entry("optimized", "timing"),
            "sdpa_only": entry("sdpa_only", "timing"),
            "sdpa": entry("sdpa", "timing"),
        },
        "correctness": {
            "pooling_only_vs_baseline": entry("pooling_only", "correctness"),
            "optimized_vs_baseline": entry("optimized", "correctness"),
            "sdpa_only_vs_baseline": entry("sdpa_only", "correctness"),
            "sdpa_vs_baseline": entry("sdpa", "correctness"),
        },
    }
    if input_kind == "synthetic":
        result["input"]["note"] = (
            "Synthetic random features; results are not brain-accuracy measurements."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"device={device} batches={len(batches)} steps={args.steps} "
        f"warmup={args.warmup} order={order}"
    )
    for name in order:
        timing = entry(name, "timing")
        if timing is None:
            continue
        if timing.get("enabled") is False:
            print(f"{name}: skipped ({timing['reason']})")
            continue
        conversion = entry(name, "conversion")
        extra = "" if conversion is None else f", conversion {conversion['time_ms']:.3f} ms"
        print(
            f"{name}: median {timing['median_ms']:.3f} ms "
            f"(p25 {timing['p25_ms']:.3f}, p75 {timing['p75_ms']:.3f}){extra}"
        )

    failed = []
    for name, correctness in result["correctness"].items():
        if correctness is None:
            continue
        print(
            f"{name}: allclose={correctness['allclose']} "
            f"max_abs={format_optional(correctness['max_abs_error'])} "
            f"rel_l2={format_optional(correctness['relative_l2'])} "
            f"corr={format_optional(correctness['centered_correlation'])} "
            f"tolerance_failures={correctness['tolerance_failure_elements']} "
            f"nonfinite={correctness['nonfinite_candidate_elements']}"
        )
        if not correctness["allclose"]:
            failed.append(name)
    print(f"wrote {args.output}")

    if failed:
        print(f"correctness failure: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
