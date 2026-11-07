# TRIBE v2: faster feature extraction

This research derivative reduces the cost of extracting video, audio, and text
features for TRIBE v2. It uses the released models and preserves the original
extractors. The changes were developed with DeepSeek V4.1 Flash and reviewed
and measured with Codex on an AMD Radeon RX 9070 XT.

The primary experiment is feature extraction. The optional brain-readout
optimization is documented separately below. The English research report,
its TeX source, aggregate measurements, and review record accompany the code:

- [Feature-extraction report](report/features.pdf) · [TeX](report/features.tex)
- [Experiment protocol and reproduction](EXPERIMENT.md)
- [Review findings and decisions](REVIEW.md) · [Measurements](results/)
- [Secondary readout report](report/report.pdf)

## Use the optimized extractors

Install an appropriate ROCm build of PyTorch and torchvision first, using the
[official installation selector](https://pytorch.org/get-started/locally/).
ROCm uses the PyTorch device name `cuda`. Then install this checkout:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

The measured runtime is Torch 2.12.0+rocm7.14.0, torchvision
0.27.0+rocm7.14.0, and Transformers 5.14.1. CPU unit tests also pass with
Torch 2.6.0 and the same non-Torch dependencies. Transformers is pinned because
the streamlined video path uses its encoder internals. Torch upper bounds were
removed to avoid replacing the working ROCm runtime with an older build.
A clean installation on other runtime combinations is not established. The
measurements used an existing offline Hugging Face cache; feature loaders do not
pin every weight/tokenizer/processor revision, so new downloads can differ.

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder="./efficient-cache",
    device="cuda",
    config_update={
        "data.video_feature.name": "EfficientVJEPA2Video",
        "data.video_feature.compute_dtype": "float16",
        "data.audio_feature.name": "EfficientWav2VecBert",
        "data.text_feature.name": "EfficientHuggingFaceText",
    },
)
events = model.get_events_dataframe(video_path="path/to/video.mp4")
predictions, segments = model.predict(events=events)
```

The model/tokenizer access requirements are inherited from upstream. Extractor
class names separate their caches from the original classes. The video precision
setting also participates in the cache identity. Existing caches are not
rewritten; a cache hit does not measure fresh extraction speed.

The example selects the validated 64-frame FP16 path. The extractor class itself
defaults to FP32; remove the `compute_dtype` override to use it. FP16 keeps FP32
token means but changes feature values. All 672 existing subject/movie pairs
pass the fixed downstream prediction margins; this is not feature equality.

## What changes

- **Video:** remove the unused V-JEPA 2 predictor, retain selected token means
  instead of every hidden state, and cache preprocessed frames at identical
  timestamps within each event. All 40 encoder layers still run.
- **Text:** transfer only the required layers and target-word tokens to the CPU;
  disable the unused generation cache. Preserve the original batching and CPU
  token mean, including its BF16 rounding.
- **Audio:** apply the original nearest interpolation before transferring
  features to the CPU, and release native-rate states before the next event.
  Preserve all 25 cached layers used by the original audio implementation.

These paths target the released V-JEPA 2, Wav2Vec2-BERT, and Llama-3.2-3B
configuration. Video verification uses fixed-resolution inputs. The predictor
removal reduces loaded video parameters by 22,381,312 (about 2.16%); it does not
reduce the model download size. FP16 separately reduces encoder weight storage.

## Measurements

Results are reported as uncached workload timings, with model loading included for video
and excluded for text/audio. Transcription is outside these timings. Individual
repetitions and peak PyTorch tensor allocation are retained in `results/`.

The FP32 video rewrite passes the feature tolerance in both alternating-order
comparisons: 241.81/242.40 s versus 197.04/197.08 s (1.23x). Peak tensor allocation
decreases from 9.47 to 4.52 GiB. The separate FP16 candidate takes 21.36/21.01 s
and 2.26 GiB: **11.43x faster and 76.16% less peak allocation** than the original
FP32 workload. Its prediction regression passes all 672 subject/movie pairs.

Text extraction on 128 words is bitwise equal to the original: 13.05/15.48 s
versus 6.27/6.31 s (two alternating-order repetitions). Its ratio of median
times is 2.27. Audio extraction on four 60-second events is also bitwise equal:
4.67/4.61 s versus 4.08/4.09 s (1.14x by the ratio of medians), with the same
peak tensor allocation in both conditions after the lifetime correction.

Prediction regression checks one initial subject and four movies, then all
168 subjects with complete existing BOLD caches. Every one of the 672 pairs
passes the same fixed margins. The largest absolute mean parcel-r change is
6.51e-5; the largest parcel-r change is 7.88e-4; the largest absolute relative
vertex-MSE change is 3.19e-5. It does not establish equivalence on an
independently held-out population or unseen stimuli. The feature gate and prediction margins are specified separately in
[EXPERIMENT.md](EXPERIMENT.md).

## Reproduce fresh extraction

Run from this checkout with a local event table containing the original four
movies, audio events, and contextualized words:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m benchmarks.benchmark_features \
  --events /path/to/events.parquet --modality video \
  --repeats 2 --output results/features_video_float32.json \
  --save-features .work/video-reference.pt
python -m benchmarks.benchmark_features \
  --events /path/to/events.parquet --modality text \
  --repeats 2 --output results/features_text.json
python -m benchmarks.benchmark_features \
  --events /path/to/events.parquet --modality audio \
  --repeats 2 --output results/features_audio.json
python -m benchmarks.benchmark_video_precision \
  --events /path/to/events.parquet --baseline-features .work/video-reference.pt \
  --repeats 2 --output results/features_video_float16.json
```

The FP16 command records timing and feature drift; it does not assert brain
accuracy. Run GPU benchmarks sequentially. Full prediction-regression commands
and the historical-cache validation are in [EXPERIMENT.md](EXPERIMENT.md).
Stimuli, transcripts, recordings, extracted arrays, and weights are not included.

## Optional brain-readout optimization

`TribeModel.from_pretrained(..., optimize=True)` fuses the released
average-subject affine readout and pools before that readout. It reduces the
brain model from 177,205,397 to 156,492,437 parameters (11.69%). On the measured
ROCm batch-8 workload, median forward time decreases from 42.91 to 35.69 ms
(1.20x); reversing condition order gives 1.18x. All four movie inputs pass the
numerical regression gates. This does not measure feature extraction.

The conversion is inference-only. Converted state dictionaries are incompatible
with the original configuration; reload the original checkpoint to train or
restore intermediate readout hooks. The measured 553 ms conversion cost takes
about 77 batch-8 calls to amortize, so a total-time benefit for one four-movie
pass is not established. See the [readout report](report/report.pdf).

This derivative retains the upstream CC-BY-NC-4.0 license and attribution.
The original project documentation follows.

---

<div align="center">

# TRIBE v2

**A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

📄 [Paper](https://arxiv.org/abs/2605.04326)  |  ▶️ [Demo](https://aidemos.atmeta.com/tribev2/) | 🤗 [Weights](https://huggingface.co/facebook/tribev2)

</div>

TRIBE v2 is a deep multimodal brain encoding model that predicts fMRI brain responses to naturalistic stimuli (video, audio, text). It combines state-of-the-art text, audio and video models into a unified Transformer architecture that maps multimodal representations onto the cortical surface.

## Quick start

Load a pretrained model from HuggingFace and predict brain responses to a video:

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)
print(preds.shape)  # (n_timesteps, n_vertices)
```

Predictions are for the "average" subject (see paper for details) and live on the **fsaverage5** cortical mesh (~20k vertices).
They are offset by 5 seconds in the past, in order to compensate for the hemodynamic lag.

You can also pass `text_path` or `audio_path` to `model.get_events_dataframe` — text is automatically converted to speech and transcribed to obtain word-level timings.

For a full walkthrough with brain visualizations, see the [Colab demo notebook](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb).

## Installation

**Basic** (inference only):
```bash
pip install -e .
```

**With brain visualization**:
```bash
pip install -e ".[plotting]"
```

**With training dependencies** (PyTorch Lightning, W&B, etc.):
```bash
pip install -e ".[training]"
```

## Training a model from scratch

### 1. Set environment variables

Configure data/output paths and Slurm partition (or edit `tribev2/grids/defaults.py` directly):

```bash
export DATAPATH="/path/to/studies"
export SAVEPATH="/path/to/output"
```


### 2. Run training

**Local test run:**
```bash
python -m tribev2.grids.test_run
```

**Grid search on Slurm:**
```bash
python -m tribev2.grids.run_cortical
python -m tribev2.grids.run_subcortical
```

## Project structure

```
tribev2/
├── main.py              # Experiment pipeline: Data, TribeExperiment
├── model.py             # FmriEncoder: Transformer-based multimodal→fMRI model
├── pl_module.py         # PyTorch Lightning training module
├── demo_utils.py        # TribeModel and helpers for inference from text/audio/video
├── eventstransforms.py  # Custom event transforms (word extraction, chunking, …)
├── utils.py             # Multi-study loading, splitting, subject weighting
├── utils_fmri.py        # Surface projection (MNI / fsaverage) and ROI analysis
├── grids/
│   ├── defaults.py      # Full default experiment configuration
│   └── test_run.py      # Quick local test entry point
├── plotting/            # Brain visualization (PyVista & Nilearn backends)
└── studies/             # Dataset definitions (Algonauts2025, Lahner2024, …)
```

## Contributing to open science

If you use this software, please share your results with the broader research community using the following citation:

```bibtex
@article{dascoli2026foundation,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brooks, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  journal={arXiv preprint arXiv:2605.04326},
  year={2026}
}
```

## License

This project is licensed under CC-BY-NC-4.0. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.
