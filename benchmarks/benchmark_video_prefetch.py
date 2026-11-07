#!/usr/bin/env python
"""Compare production video prefetch with a serial CPU-read control.

The baseline defers submitted reads until result() on the main thread, matching
serial CPU-read/GPU-forward ordering without a worker. The candidate runs the
production extractor unchanged. One 60-second event, three alternating pairs;
model loading is included. This is not a downstream brain-accuracy evaluation.
"""

import argparse
import gc
import json
from functools import partial
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from neuralset.extractors.image import HuggingFaceImage

from benchmarks.benchmark_features import (
    compare_features,
    run_raw,
    select_events,
)
from tribev2 import feature_extractors
from tribev2.feature_extractors import EfficientVJEPA2Video


class SerialFrameReader:
    """Benchmark control: defer each CPU read until the main thread requests it."""

    def __init__(self, *, max_workers):
        assert max_workers == 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, function, *args):
        # Production consumes each handle once; this control starts no worker.
        return SimpleNamespace(result=partial(function, *args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events, selection = select_events(pd.read_parquet(args.events), "video", video_duration=60)
    # Use a steady 60-second event; four-second samples overrepresent clamped startup frames.
    events, selection = events[:1], selection[:1]
    common = dict(cache_n_layers=20, layers=[.5, .75, 1.], layer_aggregation="group_mean")
    model_name = "facebook/vjepa2-vitg-fpc64-256"

    def make():
        image = HuggingFaceImage(
            model_name=model_name, device="cuda",
            infra={"keep_in_ram": False}, **common,
        )
        return EfficientVJEPA2Video(frequency=2, clip_duration=4, image=image,
                   compute_dtype="float16", num_frames=64)

    baseline = make()
    candidate = make()
    warm = [events[0].model_copy(update={"duration": .5})]

    result = {
        "status": "running", "modality": "video", "num_frames": 64,
        "torch": torch.__version__, "hip": torch.version.hip,
        "transformers": version("transformers"), "neuralset": version("neuralset"),
        "device": torch.cuda.get_device_name(), "model": model_name,
        "model_dtype": "torch.float16", "mean_dtype": str(torch.float32),
        "selection": selection, "cache_bypassed": True, "model_load_in_timing": True,
        "baseline_mode": "serial_deferred_cpu_reads",
        "candidate_mode": "production_single_window_prefetch",
        "criterion": "finite bitwise equality in all 3 pairs and candidate median <= 0.95 baseline",
        "runs": [], "comparisons": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with patch.object(feature_extractors, "ThreadPoolExecutor", SerialFrameReader):
            run_raw(baseline, warm, "video", "cuda")
        run_raw(candidate, warm, "video", "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        for repeat in range(3):
            order = ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
            features = {}
            for name in order:
                if name == "baseline":
                    with patch.object(feature_extractors, "ThreadPoolExecutor", SerialFrameReader):
                        outputs, row = run_raw(baseline, events, "video", "cuda")
                else:
                    outputs, row = run_raw(candidate, events, "video", "cuda")
                features[name] = outputs
                result["runs"].append({"repeat": repeat, "condition": name, **row})
                print(name, "repeat", repeat, "seconds", row["wall_s"], flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            comparison = compare_features(features["baseline"], features["candidate"])
            result["comparisons"].append({"repeat": repeat, **comparison})
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            if not comparison["bitwise_equal"] or not comparison["finite"]:
                raise AssertionError("Features are not finite and bitwise equal")
        base_times = [r["wall_s"] for r in result["runs"] if r["condition"] == "baseline"]
        cand_times = [r["wall_s"] for r in result["runs"] if r["condition"] == "candidate"]
        base_median = float(np.median(base_times))
        cand_median = float(np.median(cand_times))
        result["ratio_of_median_times"] = cand_median / base_median
        result["median_times_s"] = {"baseline": base_median, "candidate": cand_median}
        result["accepted"] = bool(
            all(c["bitwise_equal"] for c in result["comparisons"])
            and len(result["comparisons"]) == 3
            and cand_median <= 0.95 * base_median
        )
        result["status"] = "measured"
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
