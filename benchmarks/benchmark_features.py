"""Compare fresh extraction on four movies; no stimulus content enters JSON.

Video: one independent four-second event per movie at 120/300/480/660 seconds.
Text: words 100:116 and 800:816 per movie, in unchanged batches of four.
Audio: the third audio event in each movie. Feature cache reads are bypassed.
Video timing includes model loading; text/audio timing excludes model loading.
"""

import argparse
import gc
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralset.base import Frequency
from neuralset.events.etypes import Audio, Video, Word
from neuralset.extractors.audio import Wav2VecBert
from neuralset.extractors.image import HuggingFaceImage
from neuralset.extractors.text import HuggingFaceText
from neuralset.extractors.video import HuggingFaceVideo
from transformers import AutoConfig

from tribev2.audio_features import EfficientWav2VecBert
from tribev2.feature_extractors import EfficientVJEPA2Video
from tribev2.text_features import EfficientHuggingFaceText


def select_events(table, modality, video_duration=4):
    event_type = {"video": "Video", "text": "Word", "audio": "Audio"}[modality]
    groups = list(table[table.type == event_type].groupby("run", sort=False))
    if len(groups) != 4:
        raise ValueError("Expected exactly four movie runs")
    events, selection = [], []
    for ordinal, (_, group) in enumerate(groups):
        if modality == "video":
            offset = [120.0, 300.0, 480.0, 660.0][ordinal]
            covered = (group.offset <= offset) & (group.offset + group.duration >= offset + video_duration)
            if group.filepath.nunique() != 1 or not covered.any():
                raise ValueError("Selected video interval is not covered by the event table")
            row = group.iloc[0]
            events.append(Video(filepath=row.filepath, frequency=row.frequency,
                                offset=offset, duration=video_duration, start=0, timeline="sample"))
            selection.append({"movie_ordinal": ordinal, "offset_s": offset, "duration_s": video_duration})
        elif modality == "text":
            for begin in (100, 800):
                rows = group.iloc[begin:begin + 16]
                if len(rows) != 16 or rows[["text", "context"]].isna().any().any():
                    raise ValueError("Missing word samples or context")
                events.extend(Word(text=r.text, context=r.context, start=r.start,
                                   duration=r.duration, timeline="sample") for r in rows.itertuples())
                selection.append({"movie_ordinal": ordinal, "first_word": begin, "n_words": 16})
        else:
            row = group.iloc[2]
            events.append(Audio(filepath=row.filepath, offset=row.offset, start=row.start,
                                duration=row.duration, frequency=row.frequency, timeline="sample"))
            selection.append({"movie_ordinal": ordinal, "event_index": 2, "duration_s": row.duration})
    return events, selection


def compare_features(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Missing or unequal output counts")
    maximum, squared, elements, failures = 0.0, 0.0, 0, 0
    equal, finite = True, True
    for ref, new in zip(reference, candidate):
        if ref.shape != new.shape or ref.dtype != new.dtype:
            raise ValueError("Feature shape or dtype differs")
        valid = bool(np.isfinite(ref).all() and np.isfinite(new).all())
        finite = finite and valid
        equal = equal and bool(np.array_equal(ref, new))
        if valid:
            delta = new.astype(np.float64) - ref.astype(np.float64)
            maximum = max(maximum, float(np.abs(delta).max()))
            squared += float(np.square(delta).sum())
            failures += int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(ref)))
        else:
            failures += ref.size
        elements += ref.size
    return {"passed": finite and failures == 0, "finite": finite,
            "bitwise_equal": equal, "max_abs": maximum if finite else None,
            "rms_error": (squared / elements) ** .5 if finite else None,
            "failed_elements": failures, "compared_elements": elements,
            "rtol": 1e-4, "atol": 1e-5}


def run_raw(extractor, events, modality, device):
    # Calling the decorated method would measure Exca RAM/disk cache hits.
    raw = extractor.infra._infra_method.method
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs = list(raw(extractor, events))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    if modality == "text":
        arrays, timing = outputs, None
    else:
        arrays = [np.asarray(output.data) for output in outputs]
        timing = [{"frequency": float(output.frequency), "duration": output.duration}
                  for output in outputs]
    if len(arrays) != len(events):
        raise ValueError("Extractor dropped events")
    widths = {"video": (20, 1408), "text": (20, 3072), "audio": (25, 1024)}
    for array, event in zip(arrays, events):
        expected = widths[modality]
        if modality != "text":
            expected += (Frequency(2).to_ind(event.duration),)
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError("Output does not match the released feature dimensions")
    return arrays, {"wall_s": elapsed, "peak_allocated_bytes": peak,
                    "shapes": [list(a.shape) for a in arrays],
                    "output_dtype": str(arrays[0].dtype), "output_timing": timing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--modality", choices=["video", "text", "audio"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-features", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    events, selection = select_events(pd.read_parquet(args.events), args.modality)
    common = dict(cache_n_layers=20, layers=[.5, .75, 1.], layer_aggregation="group_mean")
    if args.modality == "video":
        model_name = "facebook/vjepa2-vitg-fpc64-256"
        def make_video(cls):
            image = HuggingFaceImage(model_name=model_name, device=args.device,
                                    infra={"keep_in_ram": False}, **common)
            return cls(frequency=2, clip_duration=4, image=image)
        extractors = {"baseline": make_video(HuggingFaceVideo),
                      "optimized": make_video(EfficientVJEPA2Video)}
        warm = [events[0].model_copy(update={"duration": .5})]
        load_s, model_dtype = None, "torch.float32"
    else:
        started = time.perf_counter()
        if args.modality == "text":
            model_name = "meta-llama/Llama-3.2-3B"
            original = HuggingFaceText(model_name=model_name, device=args.device, batch_size=4, **common)
            optimized = EfficientHuggingFaceText(device=args.device, batch_size=4, **common)
            optimized._model = original.model
            optimized._tokenizer = original.tokenizer
            optimized._pad_id = original._pad_id
            warm = events[:4]
        else:
            model_name = "facebook/w2v-bert-2.0"
            original = Wav2VecBert(device=args.device, frequency=2, **common)
            optimized = EfficientWav2VecBert(device=args.device, frequency=2, **common)
            optimized._model = original.model
            optimized._feature_extractor = original.feature_extractor
            warm = events[:1]
        if args.device == "cuda":
            torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        model_dtype = str(next(original.model.parameters()).dtype)
        extractors = {"baseline": original, "optimized": optimized}
    result = {"status": "running", "modality": args.modality,
              "torch": torch.__version__, "hip": torch.version.hip,
              "transformers": version("transformers"), "neuralset": version("neuralset"),
              "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
              "model": model_name, "separately_resolved_config_revision": AutoConfig.from_pretrained(model_name)._commit_hash,
              "model_dtype": model_dtype, "selection": selection,
              "model_load_in_timing": args.modality == "video", "separate_model_load_s": load_s,
              "cache_bypassed": True, "warmup": {}, "runs": [], "comparisons": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, extractor in extractors.items():
            _, row = run_raw(extractor, warm, args.modality, args.device)
            result["warmup"][name] = row
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        for repeat in range(args.repeats):
            order = ["baseline", "optimized"] if repeat % 2 == 0 else ["optimized", "baseline"]
            features, rows = {}, {}
            for name in order:
                features[name], row = run_raw(extractors[name], events, args.modality, args.device)
                rows[name] = row
                result["runs"].append({"repeat": repeat, "condition": name, **row})
                print(name, "repeat", repeat, "seconds", row["wall_s"], flush=True)
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            if rows["baseline"]["output_timing"] != rows["optimized"]["output_timing"]:
                raise ValueError("Output time metadata differs")
            comparison = compare_features(features["baseline"], features["optimized"])
            result["comparisons"].append({"repeat": repeat, **comparison})
            if args.save_features:
                torch.save({k: [torch.from_numpy(a.copy()) for a in v]
                            for k, v in features.items()}, args.save_features)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            if not comparison["passed"]:
                raise AssertionError("Feature regression gate failed")
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
