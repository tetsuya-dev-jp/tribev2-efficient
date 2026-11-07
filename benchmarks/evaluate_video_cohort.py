"""Evaluate preserved predictions on all complete cached subjects and movies.

Engineering regression over the frozen 168-subject local cohort, not a new
generalization experiment. Subject identifiers never enter the public output.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from benchmarks.evaluate_cached import RUNS, brain_scores
from benchmarks.evaluate_video_precision import GATES, score_passes

N_SUBJECTS = 168
FILE = re.compile(r"bold_fs5_(?P<subject>.+)_(?P<run>MOVIE[1-4]_(?:CC1|CC2|HO1|HO2))\.npz")


def discover(cache):
    # Match the exact RUNS ordering so row order is deterministic.
    found = {run: {} for run in RUNS}
    for path in cache.glob("bold_fs5_*.npz"):
        match = FILE.fullmatch(path.name)
        if not match or match["run"] not in found:
            continue
        subject = match["subject"]
        assert subject not in found[match["run"]], "Duplicate subject/run cache file"
        found[match["run"]][subject] = path
    subjects = [set(found[run]) for run in RUNS]
    assert all(s == subjects[0] for s in subjects), "Subject sets differ across runs"
    assert len(subjects[0]) == N_SUBJECTS, f"Expected {N_SUBJECTS} subjects, found {len(subjects[0])}"
    return sorted(subjects[0]), found


def validate_predictions(predictions, run, n_trs):
    entry = predictions.get(run)
    assert isinstance(entry, dict), "prediction run missing"
    ref, cand = entry.get("reference"), entry.get("candidate")
    assert isinstance(ref, torch.Tensor) and isinstance(cand, torch.Tensor)
    assert ref.shape == cand.shape == (n_trs, 20484), "prediction shape mismatch"
    assert ref.dtype == cand.dtype == torch.float32, "predictions must be float32"
    assert ref.device.type == "cpu" and cand.device.type == "cpu"
    assert torch.isfinite(ref).all() and torch.isfinite(cand).all()
    return ref, cand


def evaluate(run, ref, cand, path):
    with np.load(path, allow_pickle=False) as recorded:
        bold, labels = recorded["bold"], recorded["labels"]
    assert bold.shape == (ref.shape[0], 20484), "BOLD shape mismatch"
    assert labels.shape == (20484,), "label shape mismatch"
    assert np.isfinite(bold).all(), "non-finite BOLD"
    base_r, base_mse, n_vertices = brain_scores(ref.numpy(), bold, labels)
    new_r, new_mse, _ = brain_scores(cand.numpy(), bold, labels)
    mean_delta = float(new_r.mean() - base_r.mean())
    max_delta = float(np.abs(new_r - base_r).max())
    rel_mse = (new_mse - base_mse) / base_mse
    return {"run": run, "n_trs": int(ref.shape[0]), "scored_trs": int(ref.shape[0] - 5),
            "n_parcels": 360, "n_valid_vertices": n_vertices,
            "baseline_mean_parcel_r": float(base_r.mean()),
            "candidate_mean_parcel_r": float(new_r.mean()),
            "mean_parcel_r_delta": mean_delta, "max_abs_parcel_r_delta": max_delta,
            "baseline_mse": base_mse, "candidate_mse": new_mse,
            "relative_mse_change": rel_mse,
            "passed": score_passes(mean_delta, max_delta, rel_mse)}


def summarize(rows):
    summaries = []
    for run in RUNS:
        selected = [row for row in rows if row["run"] == run]
        if not selected:
            continue
        deltas = [row["mean_parcel_r_delta"] for row in selected]
        summaries.append({
            "run": run, "n_subjects": len(selected),
            "baseline_mean_parcel_r": float(np.mean([r["baseline_mean_parcel_r"] for r in selected])),
            "candidate_mean_parcel_r": float(np.mean([r["candidate_mean_parcel_r"] for r in selected])),
            "mean_parcel_r_delta": float(np.mean(deltas)),
            "max_abs_subject_mean_r_delta": float(np.max(np.abs(deltas))),
            "max_abs_parcel_r_delta": max(r["max_abs_parcel_r_delta"] for r in selected),
            "max_abs_relative_mse_change": max(abs(r["relative_mse_change"]) for r in selected),
            "subject_mean_r_delta_quantiles": np.quantile(deltas, [0,.05,.25,.5,.75,.95,1]).tolist(),
            "n_failed_subjects": sum(not r["passed"] for r in selected),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--score-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private-details", type=Path,
                        help="Optional per-subject diagnostics; keep this file private")
    args = parser.parse_args()
    result = {"status": "running", "n_subjects": N_SUBJECTS, "n_movies": len(RUNS),
              "expected_comparisons": N_SUBJECTS * len(RUNS), "n_comparisons": 0,
              "gates": GATES, "eval_lag": -5,
              "scope": "existing_cached_subjects_regression_not_independently_heldout",
              "note": "All subject/movie pairs use unchanged engineering margins; only movie aggregates are public.",
              "quantile_probabilities": [0,.05,.25,.5,.75,.95,1], "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = []
    try:
        bundle = torch.load(args.predictions, map_location="cpu", weights_only=True)
        assert bundle["gates"] == GATES and bundle["eval_lag"] == -5
        assert bundle["original_alignment_windows_bitwise_equal"] == 40
        assert bundle["baseline_validation"]["all_passed"]
        assert bundle["baseline_validation"]["n_timestamps"] == 16
        result["frames_per_clip"] = bundle["frames_per_clip"]
        predictions = bundle["predictions"]
        assert set(predictions) == set(RUNS)
        for run, n_trs in RUNS.items():
            reference, _ = validate_predictions(predictions, run, n_trs)
            with np.load(args.score_cache / f"preds_{run}.npz", allow_pickle=False) as cached:
                assert np.array_equal(cached["seg_starts"][:n_trs], np.arange(n_trs))
                torch.testing.assert_close(reference, torch.from_numpy(cached["preds"][:n_trs]),
                                           rtol=1e-4, atol=1e-5)
        subjects, files = discover(args.score_cache)
        for ordinal, subject in enumerate(subjects, 1):
            for run in RUNS:
                entry = predictions[run]
                row = evaluate(run, entry["reference"], entry["candidate"], files[run][subject])
                details.append({"subject_ordinal": ordinal, **row})
            result["n_comparisons"] = len(details)
            result["runs"] = summarize(details)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print("Completed subjects", ordinal, "/", len(subjects), flush=True)
        assert len(details) == result["expected_comparisons"]
        result["n_failed_comparisons"] = sum(not row["passed"] for row in details)
        result["status"] = "passed" if result["n_failed_comparisons"] == 0 else "failed"
    except Exception as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if args.private_details:
            args.private_details.parent.mkdir(parents=True, exist_ok=True)
            args.private_details.write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
