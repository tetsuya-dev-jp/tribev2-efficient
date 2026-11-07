"""Build reproducible HCP feature batches from cached events and features."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from tribev2.demo_utils import TribeModel

FEATURE_TYPES = ("Video", "Audio", "Word", "Text")
EXPECTED_BATCHES = 40
EXPECTED_PER_RUN = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--events-parquet", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not all((args.checkpoint / n).is_file() for n in ("config.yaml", "best.ckpt")):
        parser.error("--checkpoint must contain config.yaml and best.ckpt")
    if not args.events_parquet.is_file():
        parser.error(f"--events-parquet does not exist: {args.events_parquet}")
    if not args.feature_cache.is_dir():
        parser.error(f"--feature-cache does not exist: {args.feature_cache}")
    return args


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_parquet(path)
    events = events[events["type"].isin(FEATURE_TYPES)].copy()
    if events.empty:
        raise ValueError("no Video/Audio/Word/Text events found")
    events["run"] = events["run"].astype(str)
    if (events["run"].str.strip() == "").any():
        raise ValueError("some feature events have an empty run identity")
    events["split"] = "all"
    return events.sort_values(
        ["run", "timeline", "start"], kind="stable"
    ).reset_index(drop=True)


def configure_model(args: argparse.Namespace) -> TribeModel:
    xp = TribeModel.from_pretrained(
        args.checkpoint, cache_folder=args.feature_cache, device="cpu"
    )
    xp.data.batch_size = 1
    xp.data.num_workers = 0
    for modality in xp.data.features_to_use:
        extractor = getattr(xp.data, f"{modality}_feature")
        extractor.infra.mode = "read-only"
        image = getattr(extractor, "image", None)
        if image is not None and hasattr(image, "infra"):
            image.infra.mode = "read-only"
    return xp


def validate_batch(data: dict, model: torch.nn.Module, timesteps: int) -> None:
    subject_id = data.get("subject_id")
    if subject_id is None or subject_id.shape != (1, 1) or subject_id.dtype != torch.long:
        raise ValueError("subject_id must be a long tensor of shape (1, 1)")
    for modality, dims in model.feature_dims.items():
        if dims is None:
            continue
        if modality not in data:
            raise ValueError(f"missing feature tensor for modality {modality}")
        expected = (1, int(dims[0]), int(dims[1]), timesteps)
        if tuple(data[modality].shape) != expected:
            raise ValueError(
                f"{modality} shape {tuple(data[modality].shape)} != {expected}"
            )
        if not torch.is_floating_point(data[modality]):
            raise ValueError(f"{modality} features must be floating point")


def main() -> None:
    args = parse_args()
    events = load_events(args.events_parquet)

    timeline_to_run = {}
    for timeline, group in events.groupby("timeline", sort=False):
        runs = sorted(group["run"].unique())
        if len(runs) != 1:
            raise ValueError(f"timeline {timeline!r} maps to several runs: {runs}")
        timeline_to_run[str(timeline)] = runs[0]

    xp = configure_model(args)
    timesteps = int(round(xp.data.frequency * xp.data.duration_trs * xp.data.TR))
    loader = xp.data.get_loaders(events=events, split_to_build="all")["all"]

    batches, metadata = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch.segments) != 1:
                raise RuntimeError(
                    f"expected one segment per batch, got {len(batch.segments)}"
                )
            segment = batch.segments[0]
            run = timeline_to_run.get(str(segment.timeline))
            if run is None:
                raise ValueError(f"unknown timeline {segment.timeline!r}")
            data = {key: value.detach().cpu() for key, value in batch.data.items()}
            validate_batch(data, xp._model, timesteps)
            batches.append(data)
            metadata.append(
                {
                    "run": run,
                    "start": float(segment.start),
                    "duration": float(segment.duration),
                }
            )

    run_order = {run: i for i, run in enumerate(sorted({m["run"] for m in metadata}))}
    order = sorted(
        range(len(batches)),
        key=lambda i: (run_order[metadata[i]["run"]], metadata[i]["start"]),
    )
    batches = [batches[i] for i in order]
    metadata = [metadata[i] for i in order]

    counts = Counter(m["run"] for m in metadata)
    if (
        len(batches) != EXPECTED_BATCHES
        or len(counts) != 4
        or set(counts.values()) != {EXPECTED_PER_RUN}
    ):
        raise ValueError(
            f"expected {EXPECTED_BATCHES} batches from 4 runs x "
            f"{EXPECTED_PER_RUN}, got {len(batches)} from {dict(counts)}"
        )
    expected_runs = ["MOVIE1_CC1", "MOVIE2_HO1", "MOVIE3_CC2", "MOVIE4_HO2"]
    if list(run_order) != expected_runs:
        raise ValueError(f"unexpected movie runs: {list(run_order)}")
    for run in expected_runs:
        if [m["start"] for m in metadata if m["run"] == run] != list(range(0, 1000, 100)):
            raise ValueError(f"{run}: expected windows at 0, 100, ..., 900 seconds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batches, args.output)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "n_batches": len(batches),
                "batch_size": 1,
                "feature_timesteps": timesteps,
                "runs": list(run_order),
                "batches": metadata,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {len(batches)} batches to {args.output}")
    print(f"runs: {list(run_order)}")
    print(f"sidecar: {sidecar}")


if __name__ == "__main__":
    main()
