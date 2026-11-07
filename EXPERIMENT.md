# TRIBE v2 feature-extraction and inference experiments

Status: ongoing. The readout experiment below is complete. The user clarified
that feature extraction is the actual bottleneck; extraction profiling and
accuracy-preserving optimization are now the primary objective. Publication
waits for that work and an updated English report.

## Primary objective: feature extraction

Reduce fresh stimulus-extraction time and memory while maintaining prediction
accuracy. The user permits precision, implementation language, and architecture
changes. Mathematical equivalence is one route to acceptance; approximate
representations require a separate downstream check. The original extractors
and float32 path remain available.

The implementation uses neuralset 0.0.2 and Transformers 5.14.1. Video removes
the unused V-JEPA 2 predictor, streams selected token means, and reuses frames at
identical timestamps. Text transfers only selected layers/target tokens and
disables the unused KV cache. Audio applies the original nearest interpolation
before transferring all 25 cached layers. Audio must not be subsampled to 20
layers: that would change the original group means.

Fresh text/audio extraction has passed two alternating-order comparisons:

- Text: 128 words in unchanged four-word batches. Baseline 13.05/15.48 s;
  optimized 6.27/6.31 s. Both comparisons are bitwise equal over 7,864,320 values.
- Audio: four 60-second events. Baseline 4.67/4.61 s; optimized 4.08/4.09 s.
  Both comparisons are bitwise equal over 12,288,000 values.

These timings exclude model loading and transcription, and include input
processing, forward computation, and CPU transfer. The public benchmark bypasses
Exca feature caching. Video's separate short-event comparison includes model
loading; it must not be interpreted as sustained full-movie throughput.

## Completed 64-frame extraction results

The FP32 video rewrite passes both numerical comparisons: baseline
241.81/242.40 s versus 197.04/197.08 s, a 1.23x ratio of medians. Peak PyTorch
allocation falls from 9.47 to 4.52 GiB. The FP16 path takes 21.36/21.01 s and
2.26 GiB: 11.43x faster and 76.16% lower peak allocation than the original FP32
short-event workload. FP16 timings were recorded separately, not interleaved
with the original FP32 runs.

The full 64-frame FP16 prediction comparison passes all 672 subject/movie pairs
across 168 existing subjects and four movies. Worst absolute changes are
6.51e-5 for mean parcel r, 7.88e-4 for a parcel r, and 3.19e-5 for relative
labeled-vertex MSE. All are below the unchanged margins below. This supports
adoption for this evaluated workflow, not generalization to unseen inputs.

## Prespecified approximate-video evaluation

Before reading candidate brain scores, fix the candidate to float16 encoder
weights/computation with float32 token means, 64 frames per four-second window,
2 Hz outputs, and original CPU preprocessing. Do not change the trained encoder
architecture, frame sampling, event boundaries, or layer groups in this
candidate. Original text/audio features and the original, unfused brain model
are held fixed.

Use the original 60 video events across four movies, including their unequal
final durations. Reconstruct 40 original 100-second windows with neuralset's
event alignment and retain final zero padding. Verify that reconstruction from
the original cache is bitwise identical to the frozen baseline batches.
Current original float32 re-extraction matches the historical cache bitwise at
16 checked timestamps across all four movies, including variable-duration tails.
This is a provenance spot check, not a fresh full-movie float32 baseline.

Score one existing evaluation subject with the canonical lag -5 and valid movie
lengths 921/918/915/901. For **each movie**, require:

- Absolute mean parcel Pearson-r change at most `1e-4`.
- Maximum absolute change among 360 parcel correlations at most `1e-3`.
- Absolute relative change of labeled-vertex MSE at most `1e-3`.

Report signed changes and prediction errors. These are engineering regression
margins, not a statistical noninferiority test or population-equivalence claim.
They are fixed before the candidate scores and must not be loosened afterward.
The numerical feature gate (`atol=1e-5`, `rtol=1e-4`) remains mandatory for the
float32 rewrite, but is not the acceptance rule for the approximate candidate.

Before obtaining candidate scores, an inventory identified 168 existing
subjects with all four BOLD movie caches (672 subject/movie pairs). The final
precision decision therefore also requires every one of those pairs to pass
the same margins. No margin is changed. Reuse the original and candidate
average-subject predictions; only the observed BOLD changes across subjects.
The cohort script requires identical complete subject sets across movies and
uses the canonical scoring function. Public outputs contain four movie-level
aggregates, worst-case changes, and subject-change quantiles. Individual
diagnostics and prediction tensors stay private. This broader check remains an
engineering regression on existing cached subjects, not an independently held-out
population study. The original single-subject comparison is retained as an
alignment/provenance check.

Exploratory video measurements favor float16 with float32 means: 0.555 s per
clip versus about 5.95 s for the streamlined float32 encoder, with relative
feature L2 error `1.61e-4` on one sampled clip. This is a kernel probe, not yet a
prediction-accuracy result. Larger batches did not improve per-clip time.
Compilation did not meaningfully accelerate the float16 path and introduced
additional numerical differences, so eager execution is retained. Reducing
frames to 32/16 and moving preprocessing to the GPU are separate exploratory
candidates; neither is adopted from its speed alone.

### Additional candidates fixed before their score results

If the 64-frame FP16 path passes all 672 subject/movie pairs, evaluate 32 frames at the
same 4-second window width and 2 Hz output rate. Preserve the original checkpoint,
event boundaries, preprocessing, layer groups, and all three score margins.
First score Movie 1 as a rejection screen. Any failed margin rejects this
candidate; passing that screen only triggers the remaining full four-movie
evaluation, followed by the same 168-subject checks. Use a separate frame-count
cache version. This screen is not an
independent generalization experiment, and must never be reported as full acceptance.

One-window CPU preprocessing prefetch is a separate exact-input candidate.
Its acceptance requires bitwise-equal extracted features and at least 5% lower
median uncached time in three alternating paired comparisons on one 60-second
event (Movie 1, offset 120 s; 120 clips). This workload was selected before the
prefetch measurements to represent steady windows rather than mostly clamped
startup frames; model loading is included in both conditions. Its estimated ceiling is
small because a single FP16 encoder call takes about 0.55 s versus roughly
0.61 s including processing on steady 60-second events. Keeping processed CPU
frames in FP16 is lower priority: it saves only about 24 MiB of host cache,
not the dominant encoder GPU storage, and has not been adopted.

## Reduced-frame decision

Reject the 32-frame candidate under the fixed regression margins. Its first-movie
screen increased mean parcel r by 0.002229 but changed individual parcel r by
-0.041919 to +0.036098, exceeding the 0.001 regional margin. Relative vertex MSE
changed by -0.0009245. A cached repeat with signed diagnostics reproduced the
original screen scores exactly. Aggregate improvement is not evidence of
preserved regional accuracy. This is a rejection screen on one existing subject
and one movie, not a full four-movie or 168-subject evaluation.

## Secondary readout objective and acceptance criteria

Reduce the inference cost of the released TRIBE v2 brain encoder without retraining
or changing stimulus representations. Preserve the original path by default and
provide an explicit inference-only conversion. Original checkpoints and training
remain supported on the original path. Converted models cannot resume training.

Accept a candidate only after numerical comparison with the unmodified model,
targeted contract tests, synchronized accelerator timing, and independent review.
The initial float32 numerical gate is `rtol=1e-4, atol=1e-5`, including finite outputs.
Report brain-response accuracy separately from numerical agreement and synthetic
timing. Do not treat a correlation alone as evidence of numerical equivalence.

## Fixed sources

- Upstream: facebookresearch/tribev2, commit
  `af58661791a351a448a489042a28f6c37e1c14b7`.
- Released checkpoint: facebook/tribev2 revision
  `f894e783020944dcd96e5568550afe2aa9743f9f`, `best.ckpt`.
- Checkpoint dimensions: text `(2,3072)`, audio `(2,1024)`, video `(2,1408)`;
  200 input steps, 100 output steps, hidden width 1152, readout width 2048,
  20,484 output vertices. The released predictor has one shared subject row.
- Available hardware: AMD Radeon RX 9070 XT, 16 GB; Intel Core i5-13500.
- Research runtime: PyTorch `2.12.0+rocm7.14.0`, HIP `7.14.60850`, Python 3.12.
  This locally installed runtime is outside the upstream Torch `<2.7` constraint;
  portability must be checked separately.
- Real inputs: existing frozen HCP movie feature caches, four original-version
  movies. Inputs, recordings, subject identifiers, and checkpoints stay local.

## Candidates and scope

1. Move adaptive temporal averaging before the affine output projection.
   Averaging commutes with an affine map because each output bin has weights
   summing to one, including unequal or overlapping bins.
2. Compose the two linear readout matrices into one map, removing the expanded
   intermediate representation and redundant parameters from converted models.
3. Evaluate PyTorch scaled-dot-product attention through the existing dependency's
   switch; accept only if measured performance and numerical gates justify it.
4. After the eager readout comparison, probe the installed PyTorch compiler on
   the same frozen input. Record compilation cost and failures separately; do not
   make compilation a dependency or infer portable support from one machine.

No pruning, quantization, feature extractor replacement, or retraining is adopted
without a separate validation dataset and evidence of maintained accuracy.
Timing the brain encoder excludes stimulus decoding, transcription, feature
extraction, and cache I/O. End-to-end speedup must not be inferred from this timing.

Implementation and benchmark generation use DeepSeek V4.1 Flash through OpenCode
Go. Codex reviews and runs the results. Private agent transcripts remain local;
the final report records review findings and dispositions without raw logs.

## Review and stopping rule

Review mathematical equivalence, output/subject semantics, training boundaries,
checkpoint compatibility, benchmark fairness, and claim-to-evidence alignment.
Fix reproducible defects and rerun affected checks. Stop this bounded experiment
when the tested candidates have decisions, verification passes, and no actionable
review finding remains. This is not a proof that no future improvement exists.

## Secondary readout results and decisions

Table 1. Median forward latency, excluding transfers and feature extraction.
Ratios compare the original and fused model within each run, not across runtimes.

| Runtime | Baseline (ms) | Fused (ms) | Baseline / fused |
| --- | ---: | ---: | ---: |
| ROCm, batch 1 | 11.31 | 10.22 | 1.11 |
| ROCm, batch 8 | 42.91 | 35.69 | 1.20 |
| ROCm, batch 8, reverse order | 48.68 | 41.30 | 1.18 |
| CPU Torch 2.6, batch 1 | 163.67 | 133.47 | 1.23 |

Batch-8 peak PyTorch GPU allocation decreases from 1010.19 to 919.85 MiB
(8.94%). This includes tensor allocations for inputs; it is not physical VRAM.
Parameters decrease from 177,205,397 to 156,492,437 (11.69%, about 79 MiB in
float32). The upstream checkpoint on disk is unchanged.

In-place conversion takes 553 ms in the main GPU batch-8 run, with a sampled
incremental process RSS of 533.80 MiB. Conversion temporarily retains original
and composed weights. Its time divided by the per-call saving gives about 77
batch-8 calls to amortize conversion. Five calls cover the four-movie input set,
so this experiment does not show a total-runtime advantage for one short pass.
The production loader converts on CPU before transfer; cold-loader peak memory
and total loading time were not measured. Reuse favors the forward-time savings.

The reversed-order run has higher absolute latency for every condition. No cause
was established; the retained conclusion is a 1.18--1.20x within-run speed ratio
at batch 8, not a universal millisecond latency or a confidence interval.

- **Adopt affine fusion with early pooling as an opt-in conversion.** Both changes
  contribute to lower measured forward cost. Original training and checkpoint
  behavior remain the default. Model reuse amortizes conversion.
- **Do not enable SDPA by default.** On this GPU it slows both the original and
  fused model, though fused + SDPA uses less peak tensor allocation. CPU timing
  does not show a meaningful extra benefit in the final run.
- **Keep compilation as an exploratory benchmark.** Compared with fused eager
  inference, its observed steady ratios are about 1.15 at batch 1 and 1.11 at
  batch 8. Startup is seconds, cache-dependent, and tested in fixed eager-first
  order. It is not recommended for the short four-movie workload.
- **Defer approximation and representation changes.** Pruning, quantization,
  distillation, and extractor replacement cannot meet the accuracy requirement
  from these numerical regression checks alone; they need independent validation.

All 40 frozen inputs pass elementwise finite-output and tolerance gates for every
reported eager condition. In the pooled/unpooled GPU evaluation, maximum absolute
errors are 4.77e-6 and 5.60e-6. For one existing subject across four movies, the
maximum change in mean parcel correlation is 1.29e-9, the maximum change in any
parcel is 3.56e-8, and the largest absolute relative labeled-vertex MSE change is
2.68e-10. The canonical lag and valid-length trimming are preserved. This is
numerical regression evidence, not a new population-level accuracy result.

Readout-phase verification:

- 30 tests pass with the research ROCm installation and with CPU Torch 2.6.0;
  the unit tests themselves run on CPU. Dependency deprecation warnings remain.
- The cache-preparation script reproduces all 40 evaluated input tensors exactly.
- Four five-condition eager benchmark runs and two compiler probes pass their
  numerical gates; pooled/unpooled and brain-score checks pass on all four movies.
- The English report's tables and figures are regenerated from checked aggregate
  JSON; the PDF build and visual inspection are completed before publication.
- DeepSeek drafts and revisions receive Codex review; DeepSeek also reviews the
  final conversion and report, alongside an independent Codex review. Concrete
  findings and dispositions are in [REVIEW.md](REVIEW.md).

The tested candidates now have decisions. No further production defect remains
from the bounded reviews. This does not establish an exhaustive optimum over
architectures, hardware, or inputs.

## Reproduction

Install the repository with its inference dependencies and test extra in an
appropriate environment. This derivative removes the upstream Torch/torchvision
upper bounds to retain the separately provisioned RX 9070 XT ROCm runtime and
pins Transformers 5.14.1 for the video encoder internals. Install the appropriate
ROCm wheels first. The CPU 2.6
check reused the installed non-Torch dependencies; it was not a clean install.

Use the pinned checkpoint above in a local directory containing `config.yaml`
and `best.ckpt`. The following variables denote local inputs, not public downloads:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TRIBE_CHECKPOINT=/path/to/pinned/checkpoint
HCP_EVENTS=/path/to/original-version-feature-events.parquet
TRIBE_FEATURE_CACHE=/path/to/existing/tribev2/feature-cache
HCP_EVALUATION_CACHE=/path/to/existing/hcp/evaluation-cache
HCP_SUBJECT_ID=your_existing_evaluation_subject

python -m pytest tests -q
python -m benchmarks.prepare_hcp_batches \
  --checkpoint "$TRIBE_CHECKPOINT" --events-parquet "$HCP_EVENTS" \
  --feature-cache "$TRIBE_FEATURE_CACHE" --output .work/hcp_feature_batches.pt
```

The preparation command only reads feature caches. It requires the original
four movie identities and ten windows per movie at 0, 100, ..., 900 seconds.
The saved list has 40 dictionaries in movie/time order and a JSON sidecar.
Create five batch-8 dictionaries without changing order:

```bash
python - <<'PY'
from pathlib import Path
import torch
batches = torch.load('.work/hcp_feature_batches.pt', weights_only=True)
assert len(batches) == 40
grouped = []
for start in range(0, len(batches), 8):
    grouped.append({key: torch.cat([batch[key] for batch in batches[start:start+8]])
                    for key in batches[start]})
torch.save(grouped, '.work/hcp_feature_batches_b8.pt')
PY

python -m benchmarks.benchmark_inference --checkpoint "$TRIBE_CHECKPOINT" \
  --device cuda --batches .work/hcp_feature_batches.pt \
  --steps 80 --warmup 10 --sdpa --output results/rocm_hcp_b1.json
python -m benchmarks.benchmark_inference --checkpoint "$TRIBE_CHECKPOINT" \
  --device cuda --batches .work/hcp_feature_batches_b8.pt \
  --steps 60 --warmup 10 --sdpa --output results/rocm_hcp_b8.json
python -m benchmarks.benchmark_inference --checkpoint "$TRIBE_CHECKPOINT" \
  --device cuda --batches .work/hcp_feature_batches_b8.pt \
  --steps 60 --warmup 10 --sdpa --reverse-order \
  --output results/rocm_hcp_b8_reverse.json
```

Run these sequentially with no other performance experiment active. For the CPU
comparison, use the Torch 2.6 CPU interpreter, `--device cpu`, the batch-1 file,
`--steps 40 --warmup 5 --sdpa`, and `--output results/cpu26_hcp_b1.json`.
The benchmark's default 30 calls do not cover all 40 real batch-1 windows;
the commands above cover the complete input sequence.

```bash
python -m benchmarks.evaluate_cached --checkpoint "$TRIBE_CHECKPOINT" \
  --batches .work/hcp_feature_batches.pt --cache "$HCP_EVALUATION_CACHE" \
  --subject "$HCP_SUBJECT_ID" --output results/hcp_regression.json
```

The evaluation cache must contain `encoding_latent/MOVIE*.npz.meta.json`
with valid lengths and window starts, `preds_MOVIE*.npz` with `preds` and
`seg_starts`, and `bold_fs5_SUBJECT_MOVIE*.npz` with `bold` and `labels`.
Movie names and valid lengths are explicit in the evaluator. Labels must contain
360 parcel names plus the empty label for excluded vertices. This existing local
cache is not a dataset download or a population-validation protocol.

For the separate compiler probes, run `benchmarks.benchmark_compile` with the
same checkpoint and batch files, `--steps 80` at batch 1 and `--steps 60` at
batch 8; save `results/compile_hcp_b1.json` and `results/compile_hcp_b8.json`.
The recorded runs used `TORCHINDUCTOR_COMPILE_THREADS=1` and existing caches.

```bash
python -m benchmarks.render_report
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd report/report.tex
```

Plot generation additionally requires Matplotlib; building the English PDF
requires a TeX installation with `latexmk`. Aggregate JSON is included, so report
regeneration does not require the private inputs or checkpoint.

### Feature and cohort checks

The uncached short-input extraction commands are in the README. Validate the
historical video cache, then populate a separate FP16 cache and evaluate its
predictions. The metadata sidecar below is written by `prepare_hcp_batches`.

```bash
python -m benchmarks.validate_video_cache --checkpoint "$TRIBE_CHECKPOINT" \
  --events "$HCP_EVENTS" --original-cache "$TRIBE_FEATURE_CACHE" \
  --output results/video_cache_validation.json
python -m benchmarks.evaluate_video_precision --checkpoint "$TRIBE_CHECKPOINT" \
  --events "$HCP_EVENTS" --original-cache "$TRIBE_FEATURE_CACHE" \
  --candidate-cache .work/video-fp16-cache --batches .work/hcp_feature_batches.pt \
  --batch-metadata .work/hcp_feature_batches.json \
  --score-cache "$HCP_EVALUATION_CACHE" --subject "$HCP_SUBJECT_ID" \
  --baseline-validation results/video_cache_validation.json \
  --save-predictions .work/video64-predictions.pt \
  --output results/video_precision_accuracy.json
python -m benchmarks.evaluate_video_cohort \
  --predictions .work/video64-predictions.pt --score-cache "$HCP_EVALUATION_CACHE" \
  --output results/video64_cohort_accuracy.json \
  --private-details .work/video64-cohort-details.json
python -m benchmarks.render_features_report
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd report/features.tex
```

The cohort command is CPU-only and requires the frozen complete 168-subject
cache. It validates the original predictions against the selected score cache
before scoring. The optional prediction bundle and individual diagnostics are
private intermediates and must not be committed.

For the 32-frame experiment, add `--num-frames 32 --first-movie-only` to the
precision evaluator, use a separate candidate-cache directory and result file,
and omit the prediction export during screening. A `screen_passed` result is
not acceptance: rerun without `--first-movie-only`, export the full predictions,
and apply the same cohort command. A `screen_failed` result rejects that candidate.
