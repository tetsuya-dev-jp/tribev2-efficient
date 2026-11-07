"""Numerical and brain-score regression on four frozen HCP movie inputs.

This is a regression check on an existing evaluation subject, not a new
generalization experiment. Raw features, predictions and recordings stay local.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tribev2 import TribeModel

RUNS = {"MOVIE1_CC1": 921, "MOVIE2_HO1": 918, "MOVIE3_CC2": 915, "MOVIE4_HO2": 901}
RTOL, ATOL = 1e-4, 1e-5


def errors(reference, candidate):
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all()
    torch.testing.assert_close(candidate, reference, rtol=RTOL, atol=ATOL)
    ref = reference.double()
    delta = candidate.double() - ref
    return float(delta.abs().max()), float(delta.square().sum()), float(ref.square().sum())


def brain_scores(predictions, bold, labels):
    # Canonical fixed lag -5: prediction[t] is compared with BOLD[t+5].
    prediction, observed = predictions[:-5], bold[5:]
    regions = sorted(set(labels) - {""})
    assert len(regions) == 360
    prediction_parcels = np.stack(
        [prediction[:, labels == region].mean(1, dtype=np.float64) for region in regions], 1
    )
    observed_parcels = np.stack(
        [observed[:, labels == region].mean(1, dtype=np.float64) for region in regions], 1
    )
    prediction_parcels -= prediction_parcels.mean(0)
    observed_parcels -= observed_parcels.mean(0)
    correlation = (prediction_parcels * observed_parcels).sum(0) / np.sqrt(
        (prediction_parcels ** 2).sum(0) * (observed_parcels ** 2).sum(0)
    )
    assert np.isfinite(correlation).all()
    valid = labels != ""
    residual = prediction[:, valid].astype(np.float64) - observed[:, valid]
    mse = float(np.mean(residual ** 2))
    return correlation, mse, int(valid.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batches", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--sdpa", action="store_true")
    args = parser.parse_args()
    batches = torch.load(args.batches, map_location="cpu", weights_only=True)
    assert len(batches) == 40
    for batch in batches:
        for key, width in [("text", 3072), ("audio", 1024), ("video", 1408)]:
            assert batch[key].shape == (1, 2, width, 200)

    baseline = TribeModel.from_pretrained(args.checkpoint, device=args.device)._model
    candidate = TribeModel.from_pretrained(
        args.checkpoint, device=args.device, optimize=True
    )._model
    if args.sdpa:
        from x_transformers.attend import Attend
        assert not candidate.config.encoder.causal
        assert not candidate.config.encoder.residual_attn
        attention = [module for module in candidate.modules() if isinstance(module, Attend)]
        assert attention, "No attention modules were found for the SDPA comparison"
        for module in attention:
            module.flash = True

    rows = []
    try:
        with torch.inference_mode():
            for run_index, (run, n_trs) in enumerate(RUNS.items()):
                meta = json.loads((args.cache / "encoding_latent" / f"{run}.npz.meta.json").read_text())
                assert meta["window_starts"] == list(range(0, 1000, 100))
                assert meta["n_trs"] == n_trs
                reference_chunks, candidate_chunks = [], []
                unpooled_error, unpooled_squared_error, unpooled_squared_norm = 0.0, 0.0, 0.0
                for batch in batches[run_index * 10:(run_index + 1) * 10]:
                    device_batch = SimpleNamespace(data={k: v.to(args.device) for k, v in batch.items()})
                    ref = baseline(device_batch).cpu()
                    new = candidate(device_batch).cpu()
                    errors(ref, new)
                    reference_chunks.append(ref.permute(0, 2, 1).reshape(-1, 20484))
                    candidate_chunks.append(new.permute(0, 2, 1).reshape(-1, 20484))
                    native = baseline(device_batch, pool_outputs=False).cpu()
                    optimized_native = candidate(device_batch, pool_outputs=False).cpu()
                    maximum, squared_error, squared_norm = errors(native, optimized_native)
                    unpooled_error = max(unpooled_error, maximum)
                    unpooled_squared_error += squared_error
                    unpooled_squared_norm += squared_norm
                reference = torch.cat(reference_chunks)[:n_trs]
                optimized = torch.cat(candidate_chunks)[:n_trs]
                pooled_error, pooled_squared_error, pooled_squared_norm = errors(reference, optimized)
                with np.load(args.cache / f"preds_{run}.npz", allow_pickle=False) as cached:
                    assert np.array_equal(cached["seg_starts"][:n_trs], np.arange(n_trs))
                    old = torch.from_numpy(cached["preds"][:n_trs])
                    cache_error, _, _ = errors(old, reference)
                with np.load(args.cache / f"bold_fs5_{args.subject}_{run}.npz", allow_pickle=False) as recorded:
                    bold, labels = recorded["bold"], recorded["labels"]
                assert bold.shape == reference.shape == (n_trs, 20484)
                assert labels.shape == (20484,) and np.isfinite(bold).all()
                base_r, base_mse, n_vertices = brain_scores(reference.numpy(), bold, labels)
                new_r, new_mse, _ = brain_scores(optimized.numpy(), bold, labels)
                delta_r = float(new_r.mean() - base_r.mean())
                max_parcel_delta = float(np.max(np.abs(new_r - base_r)))
                relative_mse_change = (new_mse - base_mse) / base_mse
                assert abs(delta_r) < 1e-6
                assert max_parcel_delta < 1e-6
                assert abs(relative_mse_change) < 1e-5
                rows.append({
                    "run": run, "n_trs": n_trs, "scored_trs": n_trs - 5,
                    "n_parcels": 360, "n_valid_vertices": n_vertices,
                    "baseline_mean_parcel_r": float(base_r.mean()),
                    "optimized_mean_parcel_r": float(new_r.mean()),
                    "mean_parcel_r_delta": delta_r,
                    "max_abs_per_parcel_r_delta": max_parcel_delta,
                    "baseline_mse_labeled": base_mse, "optimized_mse_labeled": new_mse,
                    "relative_mse_change": relative_mse_change,
                    "cache_preds_max_abs": cache_error,
                    "pooled_max_abs": pooled_error,
                    "pooled_relative_l2": float(np.sqrt(pooled_squared_error / pooled_squared_norm)),
                    "unpooled_max_abs": unpooled_error,
                    "unpooled_relative_l2": float(np.sqrt(unpooled_squared_error / unpooled_squared_norm)),
                })
                print(run, "pooled", pooled_error, "unpooled", unpooled_error, "delta r", delta_r, flush=True)
    except AssertionError as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": "failed", "failed_run": run, "completed_runs": rows,
            "reason": str(error) or "Numerical, alignment, or score gate failed",
        }, indent=2, allow_nan=False) + "\n")
        raise
    payload = {
        "status": "passed",
        "candidate": "fused_readout_sdpa" if args.sdpa else "fused_readout",
        "date": "2026-11-08", "torch": torch.__version__, "device": args.device,
        "n_subjects": 1, "note": "Numerical regression on an existing evaluation subject; no statistical equivalence or new generalization claim.",
        "gating": {"rtol": RTOL, "atol": ATOL, "eval_lag": -5,
                   "mean_parcel_r_delta_abs_lt": 1e-6, "max_abs_per_parcel_r_delta_lt": 1e-6,
                   "relative_mse_change_abs_lt": 1e-5},
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
