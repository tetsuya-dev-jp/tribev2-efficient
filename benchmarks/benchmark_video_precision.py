"""Time the approximate FP16 video path on the same four short movie samples.

Reference arrays must come from benchmark_features --modality video with the
same events table and --save-features. This measures feature drift and latency,
not brain-response accuracy. Baseline timing belongs to the separate benchmark.
"""

import argparse
import gc
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from neuralset.extractors.image import HuggingFaceImage

from benchmarks.benchmark_features import compare_features, run_raw, select_events
from tribev2.feature_extractors import EfficientVJEPA2Video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    events, selection = select_events(pd.read_parquet(args.events), "video")
    reference = torch.load(args.baseline_features, map_location="cpu", weights_only=True)["baseline"]
    assert len(reference) == 4 and all(x.shape == (20, 1408, 8) for x in reference)
    reference = [x.numpy() for x in reference]
    image = HuggingFaceImage(model_name="facebook/vjepa2-vitg-fpc64-256", device="cuda",
                            cache_n_layers=20, layers=[.5, .75, 1.], layer_aggregation="group_mean",
                            infra={"keep_in_ram": False})
    extractor = EfficientVJEPA2Video(frequency=2, clip_duration=4, image=image, compute_dtype="float16")
    result = {"status": "running", "torch": torch.__version__, "hip": torch.version.hip,
              "model": image.model_name,
              "transformers": version("transformers"), "neuralset": version("neuralset"),
              "device": torch.cuda.get_device_name(), "model_dtype": "torch.float16",
              "mean_dtype": str(torch.float32), "selection": selection,
              "model_load_in_timing": True, "cache_bypassed": True,
              "cpu_window_prefetch": True,
              "note": "Candidate-only short-event timings; original timing measured separately. Brain-score validation is separate.",
              "runs": [], "feature_comparisons": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        _, result["warmup"] = run_raw(extractor, [events[0].model_copy(update={"duration": .5})], "video", "cuda")
        for repeat in range(args.repeats):
            gc.collect()
            torch.cuda.empty_cache()
            outputs, timing = run_raw(extractor, events, "video", "cuda")
            comparison = compare_features(reference, outputs)
            assert comparison["finite"], "Nonfinite features"
            result["runs"].append({"repeat": repeat, **timing})
            result["feature_comparisons"].append(comparison)
            print("repeat", repeat, "seconds", timing["wall_s"], "max feature error", comparison["max_abs"], flush=True)
        result["status"] = "measured"
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
