"""Optional compiler probe; compilation is never required by the core model."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tribev2 import TribeModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    batches = torch.load(args.batches, weights_only=True, map_location="cpu")
    assert batches
    model = TribeModel.from_pretrained(args.checkpoint, device="cuda", optimize=True)._model
    timings = {}
    references = []
    maximum_error, failed_elements = 0.0, 0
    with torch.inference_mode():
        for name in ["eager", "compiled"]:
            if name == "compiled":
                runner = torch.compile(model)
            else:
                runner = model
            first_call_ms = None
            for index, batch in enumerate(batches):
                value = SimpleNamespace(data={k: v.cuda() for k, v in batch.items()})
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = runner(value)
                torch.cuda.synchronize()
                if index == 0:
                    first_call_ms = (time.perf_counter() - start) * 1000
                if name == "eager":
                    references.append(output.cpu())
                else:
                    reference = references[index]
                    result = output.cpu()
                    assert torch.isfinite(result).all()
                    torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
                    difference = (result - reference).abs()
                    maximum_error = max(maximum_error, float(difference.max()))
                    failed_elements += int((difference > 1e-5 + 1e-4 * reference.abs()).sum())
                del output, value
            # All input shapes are warmed before recording steady inference.
            samples = []
            torch.cuda.reset_peak_memory_stats()
            for index in range(args.steps):
                value = SimpleNamespace(data={k: v.cuda() for k, v in batches[index % len(batches)].items()})
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = runner(value)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
                del output, value
            timings[name] = {
                "first_call_ms": first_call_ms, "samples_ms": samples,
                "median_ms": float(np.median(samples)),
                "p25_ms": float(np.percentile(samples, 25)),
                "p75_ms": float(np.percentile(samples, 75)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
    result = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0), "threads": torch.get_num_threads(),
        "n_batches": len(batches), "batch_sizes": [b["text"].shape[0] for b in batches],
        "reference": "optimized eager; original-baseline fidelity checked separately",
        "timing_scope": "forward only; H2D and cache I/O excluded; fixed eager then compiled order",
        "first_call_note": "Compile plus first forward; includes installed compiler cache effects, not a clean-cache measurement",
        "timing": timings, "max_abs_error": maximum_error,
        "failed_elements": failed_elements, "rtol": 1e-4, "atol": 1e-5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"timing": {k: v["median_ms"] for k,v in timings.items()}, "max_abs_error": maximum_error}))


if __name__ == "__main__":
    main()
