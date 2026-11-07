"""Spot-check historical video features using the current original FP32 path.

Checks four timestamps per movie: the first output of event 2, and the first,
middle, and last outputs of event 14. Uses the ORIGINAL event time grid.
Passing is a spot check, not a proof about every value in the historical cache.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralset.base import Frequency
from neuralset.extractors.video import _HFVideoModel, _VideoImage

from tribev2 import TribeModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "events", "original-cache", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_parquet(args.events)
    groups = list(table[table.type == "Video"].groupby("run", sort=False))
    assert len(groups) == 4 and all(len(group) == 15 for _, group in groups)
    xp = TribeModel.from_pretrained(args.checkpoint, device="cpu", cache_folder=args.original_cache)
    extractor = xp.data.video_feature
    extractor.infra.mode = "read-only"
    del xp
    model = _HFVideoModel(model_name=extractor.image.model_name,
                         pretrained=extractor.image.pretrained,
                         layer_type=extractor.layer_type, num_frames=extractor.num_frames)
    model.model.cuda().eval()
    assert next(model.model.parameters()).dtype == torch.float32
    subtimes = [k / model.num_frames * extractor.clip_duration
                for k in reversed(range(model.num_frames))]
    result = {"passed": False, "torch": torch.__version__, "hip": torch.version.hip,
              "transformers": version("transformers"), "neuralset": version("neuralset"),
              "device": torch.cuda.get_device_name(), "dtype": "torch.float32",
              "model_revision": model.model.config._commit_hash,
              "scope": "16 original event-grid timestamps, not a complete cache re-extraction",
              "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for ordinal, (_, group) in enumerate(groups):
            events = extractor._get_relevant_events(group.sort_values("start"), trigger=None)
            for event_index in (2, 14):
                event = events[event_index]
                reference = list(extractor._get_data([event]))[0].data
                with event.read() as video:
                    times = np.linspace(0, video.duration,
                                        Frequency(extractor.frequency).to_ind(event.duration) + 1)[1:]
                    indices = [0, len(times) // 2, len(times) - 1] if event_index == 14 else [0]
                    for index in indices:
                        images = [_VideoImage(video=video, time=max(0., times[index] - sub)).read()
                                  for sub in subtimes]
                        if extractor.max_imsize is not None:
                            factor = max(images[0].size) / extractor.max_imsize
                            if factor > 1:
                                size = tuple(int(s / factor) for s in images[0].size)
                                images = [image.resize(size) for image in images]
                        frames = np.array([np.array(image) for image in images])
                        with torch.inference_mode():
                            states = model.predict_hidden_states(frames)[0]
                            current = extractor.image._aggregate_tokens(states).cpu().numpy()
                        ref = np.asarray(reference[:, :, index])
                        assert np.isfinite(current).all() and np.isfinite(ref).all()
                        delta = np.abs(current - ref)
                        row = {"movie_ordinal": ordinal, "event_index": event_index,
                               "sample_index": index, "n_samples": len(times),
                               "event_duration_s": event.duration,
                               "max_abs": float(delta.max()),
                               "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(ref)),
                               "failed_elements": int(np.count_nonzero(delta > 1e-5 + 1e-4 * np.abs(ref))),
                               "bitwise_equal": bool(np.array_equal(current, ref))}
                        result["rows"].append(row)
                        print(row, flush=True)
                        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                        del states, current, ref, delta
        result["passed"] = all(row["failed_elements"] == 0 for row in result["rows"])
        assert result["passed"], "Current original extraction differs from historical cache"
    except Exception as error:
        result["passed"] = False
        result["error_type"] = type(error).__name__
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
