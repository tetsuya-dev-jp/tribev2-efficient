"""Evaluate float16 video extraction on four existing HCP movies.

This is an engineering regression check on an existing evaluation subject.
The original 60 event boundaries and neuralset alignment are preserved. Feature
cache population is resumable; its elapsed time is not a fresh speed benchmark.
"""

import argparse
import gc
import json
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from neuralset.base import Frequency
from neuralset.events.etypes import EventTypesHelper

from benchmarks.evaluate_cached import RUNS, brain_scores
from tribev2 import TribeModel
from tribev2.feature_extractors import EfficientVJEPA2Video

GATES = {"abs_mean_parcel_r_delta_le": 1e-4,
         "max_abs_parcel_r_delta_le": 1e-3,
         "abs_relative_mse_change_le": 1e-3}


def overlapping(events, start, duration):
    return [event for event in events
            if event.start < start + duration and event.stop > start]


def score_passes(mean_delta, maximum_delta, relative_mse_change):
    return (abs(mean_delta) <= GATES["abs_mean_parcel_r_delta_le"]
            and maximum_delta <= GATES["max_abs_parcel_r_delta_le"]
            and abs(relative_mse_change) <= GATES["abs_relative_mse_change_le"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "events", "original-cache", "candidate-cache", "batches",
                 "batch-metadata", "score-cache", "baseline-validation", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--num-frames", type=int, choices=[32, 64], default=64)
    parser.add_argument("--first-movie-only", action="store_true",
                        help="32-frame rejection screen; never full acceptance")
    args = parser.parse_args()
    if args.first_movie_only and args.num_frames != 32:
        parser.error("--first-movie-only is only defined for the 32-frame candidate")
    if args.original_cache.resolve() == args.candidate_cache.resolve():
        parser.error("Candidate cache must be separate from the original cache")
    if not torch.cuda.is_available():
        parser.error("This experiment requires the tested GPU configuration")
    provenance = json.loads(args.baseline_validation.read_text())
    assert provenance["passed"] and len(provenance["rows"]) == 16
    assert all(row["failed_elements"] == 0 for row in provenance["rows"])
    table = pd.read_parquet(args.events)
    by_run = {run: EventTypesHelper("Video").extract(group.sort_values("start"))
              for run, group in table[table.type == "Video"].groupby("run")}
    assert set(by_run) == set(RUNS) and all(len(events) == 15 for events in by_run.values())
    events = [event for run in RUNS for event in by_run[run]]
    batches = torch.load(args.batches, map_location="cpu", weights_only=True)
    metadata = json.loads(args.batch_metadata.read_text())["batches"]
    assert len(batches) == len(metadata) == 40
    for ordinal, run in enumerate(RUNS):
        entries = metadata[ordinal * 10:(ordinal + 1) * 10]
        assert all(entry["run"] == run and entry["duration"] == 100 for entry in entries)
        assert [entry["start"] for entry in entries] == list(range(0, 1000, 100))

    xp = TribeModel.from_pretrained(args.checkpoint, device="cpu", cache_folder=args.original_cache)
    original = xp.data.video_feature
    original.infra.mode = "read-only"
    config = original.model_dump()
    assert config["frequency"] == 2 and config["clip_duration"] == 4
    assert config["num_frames"] in (None, 64)
    assert config["image"]["model_name"] == "facebook/vjepa2-vitg-fpc64-256"
    assert config["image"]["cache_n_layers"] == 20
    config["name"] = "EfficientVJEPA2Video"
    config["compute_dtype"] = "float16"
    config["image"]["device"] = "cuda"
    cache_version = "fp16-token-means-fp32-v1"
    if args.num_frames == 32:
        config["num_frames"] = 32
        cache_version += "-nf32"
    config["infra"].update(folder=str(args.candidate_cache), cluster=None,
                           mode="cached", keep_in_ram=False,
                           version=cache_version)
    candidate = EfficientVJEPA2Video(**config)
    assert candidate.infra.uid() != original.infra.uid()

    scoring_runs = list(RUNS.items())[:1] if args.first_movie_only else list(RUNS.items())
    active_events = events[:15] if args.first_movie_only else events
    result = {"status": "running", "candidate": f"video_float16_float32_token_means_nf{args.num_frames}",
              "n_subjects": 1, "n_movies": len(scoring_runs), "n_events": len(active_events),
              "scope": "rejection_screen_only" if args.first_movie_only else "four_movie_regression",
              "note": "Existing-subject regression within prespecified engineering margins; no population equivalence claim.",
              "torch": torch.__version__, "hip": torch.version.hip,
              "transformers": version("transformers"), "neuralset": version("neuralset"),
              "device": torch.cuda.get_device_name(), "frames_per_clip": args.num_frames,
              "frequency_hz": 2, "clip_duration_s": 4, "cached_layers": 20,
              "extractor_cache_version": candidate.infra.version,
              "gates": GATES, "eval_lag": -5,
              "baseline_validation": {"n_timestamps": 16, "all_passed": True,
                                      "scope": "Spot check against historical cache; not a fresh full-movie FP32 extraction.",
                                      "model_revision": provenance["model_revision"]},
              "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for batch, entry in zip(batches, metadata):
            relevant = overlapping(by_run[entry["run"]], entry["start"], entry["duration"])
            aligned = original(relevant, start=entry["start"], duration=entry["duration"])
            assert torch.equal(aligned, batch["video"][0]), "Original cache alignment differs"
        result["original_alignment_windows_bitwise_equal"] = 40
        print("Verified original cache alignment in all 40 windows", flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        count = 0
        for event, array in zip(active_events, candidate._get_data(active_events), strict=True):
            timepoints = Frequency(2).to_ind(event.duration)
            assert array.data.shape == (20, 1408, timepoints)
            assert float(array.frequency) == 2
            assert array.duration == Frequency(2).to_sec(timepoints)
            assert np.isfinite(np.asarray(array.data)).all()
            count += 1
            print("Available candidate events:", count, "/", len(active_events), flush=True)
        assert count == len(active_events)
        torch.cuda.synchronize()
        result["cache_population_elapsed_s"] = time.perf_counter() - started
        result["cache_population_note"] = "Includes cache checks and possible resumed cache hits; not a fresh throughput measurement."
        result["cache_population_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["cache_population_peak_note"] = "Observed for this population call; depends on resumed cache hits."
        gc.collect()
        torch.cuda.empty_cache()
        model = xp._model.cuda().eval()
        with torch.inference_mode():
            for ordinal, (run, n_trs) in enumerate(scoring_runs):
                references, predictions = [], []
                for index in range(ordinal * 10, (ordinal + 1) * 10):
                    batch, entry = batches[index], metadata[index]
                    data = {key: value.cuda() for key, value in batch.items()}
                    ref = model(SimpleNamespace(data=data)).cpu()
                    relevant = overlapping(by_run[run], entry["start"], entry["duration"])
                    video = candidate(relevant, start=entry["start"], duration=entry["duration"])
                    assert video.shape == (2, 1408, 200) and torch.isfinite(video).all()
                    data["video"] = video.unsqueeze(0).cuda()
                    new = model(SimpleNamespace(data=data)).cpu()
                    assert torch.isfinite(ref).all() and torch.isfinite(new).all()
                    references.append(ref.permute(0, 2, 1).reshape(-1, 20484))
                    predictions.append(new.permute(0, 2, 1).reshape(-1, 20484))
                reference = torch.cat(references)[:n_trs]
                prediction = torch.cat(predictions)[:n_trs]
                with np.load(args.score_cache / f"preds_{run}.npz", allow_pickle=False) as cached:
                    assert np.array_equal(cached["seg_starts"][:n_trs], np.arange(n_trs))
                    torch.testing.assert_close(reference, torch.from_numpy(cached["preds"][:n_trs]),
                                               rtol=1e-4, atol=1e-5)
                with np.load(args.score_cache / f"bold_fs5_{args.subject}_{run}.npz", allow_pickle=False) as recording:
                    bold, labels = recording["bold"], recording["labels"]
                assert bold.shape == reference.shape == (n_trs, 20484)
                assert labels.shape == (20484,) and np.isfinite(bold).all()
                ref_r, ref_mse, n_vertices = brain_scores(reference.numpy(), bold, labels)
                new_r, new_mse, _ = brain_scores(prediction.numpy(), bold, labels)
                mean_delta = float(new_r.mean() - ref_r.mean())
                maximum_delta = float(np.abs(new_r - ref_r).max())
                relative_mse = (new_mse - ref_mse) / ref_mse
                delta = prediction.double() - reference.double()
                row = {"run": run, "n_trs": n_trs, "scored_trs": n_trs - 5,
                       "n_parcels": 360, "n_valid_vertices": n_vertices,
                       "baseline_mean_parcel_r": float(ref_r.mean()),
                       "candidate_mean_parcel_r": float(new_r.mean()),
                       "mean_parcel_r_delta": mean_delta, "max_abs_parcel_r_delta": maximum_delta,
                       "baseline_mse": ref_mse, "candidate_mse": new_mse,
                       "relative_mse_change": relative_mse,
                       "prediction_max_abs": float(delta.abs().max()),
                       "prediction_relative_l2": float(delta.norm() / reference.double().norm()),
                       "passed": score_passes(mean_delta, maximum_delta, relative_mse)}
                result["runs"].append(row)
                print(run, "mean parcel r delta", mean_delta, "passed", row["passed"], flush=True)
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        result["status"] = "passed" if all(row["passed"] for row in result["runs"]) else "failed"
        passed = result["status"] == "passed"
        if args.first_movie_only:
            result["status"] = "screen_" + result["status"]
        if not passed:
            raise AssertionError("Prespecified brain-score regression margin exceeded")
    except Exception as error:
        if result["status"] == "running":
            result["status"] = "error"
        result["error_type"] = type(error).__name__
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
