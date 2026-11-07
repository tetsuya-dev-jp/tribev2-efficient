# Review record

Implementation drafts and revisions used DeepSeek V4.1 Flash through OpenCode
Go. Codex inspected the code and executed verification; a separate Codex reviewer
examined mathematical validity, benchmark design, evaluation alignment, and the
production conversion. Raw agent transcripts are private working material.

## Findings and dispositions

### Feature extraction

| Finding | Disposition |
| --- | --- |
| Brain-readout optimization does not address the reported extraction bottleneck. | Treat extraction as the primary experiment; retain readout results separately. |
| The V-JEPA 2 predictor runs although only encoder states are consumed. | Drop its parameters before GPU transfer and run the encoder directly. |
| V-JEPA 2 hidden-state indexing includes the embedding and excludes the final layer norm. | Match the actual Transformers implementation; verify with tiny-model tests and real inputs. |
| Retaining every token from every layer dominates memory. | Keep only selected per-layer token means; run every encoder layer. Check floating-point differences separately. |
| Overlapping windows repeat frame decoding and preprocessing. | Cache preprocessed CPU frames at exact timestamps. Four-movie checks and a real-processor unit test verify bitwise input equality. Cache scope is one event and one window. |
| Variable-size frames with `max_imsize` could change the resize reference frame. | The tested HCP inputs have fixed frame dimensions. Do not extrapolate that verification to variable-resolution input. |
| Moving text means to the GPU or casting before averaging could change bfloat16 outputs. | Transfer selected tokens/layers, then retain the original CPU mean and float32 conversion. All 128 sampled words are bitwise equal in both repetitions. |
| The audio reference caches 25 layers despite `cache_n_layers=20`. | Preserve all 25 layers; select timestamps before transfer, not layers. Four real audio events are bitwise equal in both repetitions. |
| The first audio rewrite retained native-rate states until the next event's forward, increasing GPU peak. | Release outputs after CPU transfer and before yielding; a weakref test checks release at the next forward. Rerun the measured comparison after this correction. |
| Exca can return RAM cache hits even without a cache directory. | Timing calls the undecorated extraction method for both conditions. |
| A draft benchmark used incorrect imports, sampling counts, and a cached text call. | Replace those paths with verified APIs and one video interval per movie; test selection counts and comparison failures. |
| Equal candidate/reference shapes alone could hide dropped outputs. | Check output count and expected released-model dimensions against every input event. |
| Video constructors reload weights inside extraction, unlike persistent text/audio models. | Include video loading explicitly and exclude it explicitly for text/audio; report separate timing scopes. |
| A separately loaded configuration is not direct evidence of the actual weight revision. | Name this metadata `separately_resolved_config_revision`; the provenance check records the loaded video's model revision. |
| An approximate feature may fail a numerical feature gate while preserving prediction accuracy. | Evaluate float16 separately with prespecified downstream score margins. Do not label its features equivalent. |
| Half-precision means add avoidable final-output rounding. | Accumulate and return token means in float32 while keeping the encoder in float16. |
| Short independent samples have different clamped context from full-movie events. | Preserve all original 60 event boundaries for full-movie accuracy evaluation; use short samples only for their stated benchmark. |
| Historical cache revisions were not recorded. | Verify current original float32 against 16 original-grid timestamps, including unequal-duration tails. All are bitwise equal; label this a spot check. |
| Rebuilding windows by manual column arithmetic can shift tail samples. | Use neuralset alignment, verify all 40 original windows bitwise against frozen batches, and replace only video tensors. |
| A resumable cache can mix implementations or conceal incomplete event arrays. | Use a separate directory and explicit candidate cache version; validate each event's length, frequency, duration, dimensions, and finite values. |
| Cache-population timing and allocation depend on resumed cache hits. | Label them as observations of that call; use separate uncached short-input measurements for speed comparisons. |
| A draft report renderer confused raw JSON with grouped runs and used the FP16 candidate as its own baseline. | Group conditions explicitly, compare FP16 with the measured original FP32 workload, retain both repetitions, and gate matched selection/environment/output metadata. |
| A draft renderer asserted all movies pass before results existed. | Compute acceptance from recorded score rows and fixed gates; reject partial/error results and verify lag, event count, cache spot checks, and all 40 original alignments. |
| Reduced-frame evaluation can misuse a single-movie screen as acceptance. | Label screen results separately; unchanged gates on all four movies are required for full acceptance. |
| The available BOLD cache contains 168 complete subjects, not only the initial subject. | Extend the unchanged margins to all 672 subject/movie pairs; use identical predictions and canonical scoring. Public cohort outputs contain only movie aggregates. |
| Prediction-bundle metadata alone does not verify the selected score cache. | Check original predictions and segment starts against that cache before cohort scoring. |
| A sliced prediction tensor retains its padded backing storage when saved. | Clone only the valid slice for the private prediction export. |

The independent source reviewer found no blocker for the tested fixed-resolution
HCP path. A private evaluator plumbing check deliberately supplied original
features to both arms and obtained zero score changes; it is not reported as
float16 evidence. Full float16 score evaluation is recorded separately.

DeepSeek's reciprocal review raised conditional questions about layer calls,
private model attributes, and tensor-return APIs. Inspection confirmed the
actual neuralset/Transformers contracts; the exercised tests and measurements
support those paths. Hypothetical claims that contradicted those APIs were not
accepted as defects. Audio `accelerate` handling is inherited from the reference
and is outside the tested CUDA/ROCm configuration.

A later reciprocal review questioned whether alignment assertions were inside
the evaluator's exception handler; direct inspection confirmed they are. It also
questioned using `_get_data` for cache population and the inherited `__call__`
for temporal alignment. Those are the intended separate responsibilities of
neuralset, and the original 40-window reconstruction checks the alignment
contract. These claims did not identify a defect. The private prefetch probe
keeps its frame offsets, processor, and image limit fixed within each video,
and its two-video CPU check verifies timestamp order and future consumption.
Its speed and numerical outputs still require the actual paired GPU experiment.

### Secondary brain readout

| Finding | Disposition |
| --- | --- |
| The released shared head is not the arithmetic mean of all subject heads. | Conversion selects the same dropout-subject row used by the dependency. The released checkpoint already has one row. |
| Pooling commutes with an affine map, but floating-point reassociation is not bitwise exact. | Both pooled and unpooled real outputs are compared against the original model. Unequal, equal, and upsampling bins are tested. |
| Selecting a row without cloning retains the original subject-bank storage when no intermediate projection exists. | Clone the shared row; test both values and underlying storage size. |
| Unsupported average-subject configurations can fail only after mutation. | Reject missing subject-dropout configuration before changing weights. |
| Converted weights cannot use the original state-dict layout or intermediate-feature hooks. | Keep conversion opt-in, prohibit training, document reloading and hook limitations; verify original checkpoint loading and gradients. |
| Converting after GPU transfer unnecessarily transfers the unfused weights. | Convert on CPU in `from_pretrained`, then transfer. This is an implementation property, not a measured cold-start peak claim. |
| An initial benchmark draft compared CPU reference predictions with GPU candidates. | Use the same device for numerical references and every candidate. |
| Attention and readout changes were initially combined. | Add pooling-only and SDPA-only conditions; label the combined condition explicitly. |
| Fixed condition order and batch labels could confound interpretation. | Record actual batch sizes; run a reversed-order batch-8 comparison. |
| Concatenated double-precision diagnostics consumed excessive host memory. | Accumulate comparison statistics batch by batch, including nonfinite and tolerance-failure counts. |
| Numerical failures should leave evidence. | The benchmark writes diagnostics before returning failure; the brain evaluator records the failed movie, reason, and completed rows. |
| GPU allocation is not physical VRAM or activation-only memory. | Use explicit allocator terminology, include input tensors in scope, and separately record conversion time, CUDA peak, and sampled process RSS. |
| The canonical HCP lag is -5 and some saved predictions contain trailing padding. | Match the canonical evaluation: predictions[:-5] vs BOLD[5:], after valid-length trimming; validate saved prediction agreement. |
| Mean parcel differences can cancel. | Also gate the maximum absolute parcel correlation change; report vertex MSE separately. |
| One subject cannot establish population equivalence. | Describe brain results as a numerical regression check, with no confidence intervals or significance claims. |
| Compiler first-call cost depends on caches and does not measure total time to steady state. | Keep compilation an exploratory probe, state its fixed eager-then-compiled order, and do not recommend it for the short four-movie workload. |
| Fused-readout conversion also has a one-time cost. | Report 553 ms and an in-place amortization estimate of 77 batch-8 calls; do not claim a total-runtime benefit for a single four-movie pass. |
| The report generator could accept absent comparisons or mislabeled input files. | Require every reported comparison, actual batch sizes, cached-input kind, runtime, call counts, and matching reverse-order conditions before rendering. |
| Reciprocal review questioned upsampling bins and linear-baseline support. | Check the framework's integer bin definitions and the existing unequal/equal/upsampling tests, which explicitly use `linear_baseline=True`. The same affine argument applies; no production change is warranted. |
| Reciprocal review questioned the nonzero subject-dropout requirement. | The dependency itself asserts this condition for average-subject inference. Preserve the guard and make both flags explicit in the report. |

The fixed-lag score implementation has a synthetic known-shift test. Its original
model's first-movie mean parcel correlation also agrees with the existing
canonical score to about 7.5e-9 (float64 versus legacy aggregation rounding).
The preparation script reproduces all 40 evaluated feature batches exactly.
The pooling-bin definition is also checked against the
[PyTorch 2.6 source](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/AdaptivePooling.h).
DeepSeek's final review of the Codex revisions was adjudicated against source and
tests rather than accepted automatically. Both pooled and unpooled maxima already
appear in the results; an unused summary macro was removed.

The mathematical review found no need to alter the research representation or
train new weights. The production review found no further concrete defect in the
released-model conversion. Candidate decisions and executed checks are recorded
in [EXPERIMENT.md](EXPERIMENT.md); the report is generated from aggregate results.
This review record does not claim exhaustive optimization or universal accuracy.

Readout-phase verification: 30 tests passed in each of the two tested Torch environments;
all four eager benchmark runs pass all four candidate comparisons. The final
independent review found no further actionable claim-to-evidence defect.
