"""Render the feature-extraction benchmark results and figures."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.render_report import ROOT, REPORT, scientific, table

GATES = {"abs_mean_parcel_r_delta_le": 1e-4,
         "max_abs_parcel_r_delta_le": 1e-3,
         "abs_relative_mse_change_le": 1e-3}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
HATCHES = ["", "//", "..", "xx"]


def load(name):
    path = ROOT / "results" / f"{name}.json"
    return json.loads(path.read_text())


def verify_features(data, modality, expected_runs, loading):
    assert data["status"] == "passed", f"{modality}: status {data['status']}"
    assert data["modality"] == modality
    assert data["cache_bypassed"] is True
    assert data["model_load_in_timing"] is loading
    assert len(data["comparisons"]) == 2, f"{modality}: expected 2 comparisons"
    assert all(c["passed"] for c in data["comparisons"]), f"{modality}: comparison failed"
    assert all(c["rtol"] == 1e-4 and c["atol"] == 1e-5 and c["finite"]
               and c["failed_elements"] == 0 for c in data["comparisons"])
    runs = data["runs"]
    assert len(runs) == expected_runs, f"{modality}: expected {expected_runs} runs, got {len(runs)}"
    by_condition = {"baseline": [], "optimized": []}
    for run in runs:
        assert run["condition"] in by_condition, f"{modality}: bad condition {run['condition']}"
        by_condition[run["condition"]].append(run)
        wall = run["wall_s"]
        assert np.isfinite(wall) and wall > 0, f"{modality}: bad wall_s {wall}"
        assert run["peak_allocated_bytes"] > 0, f"{modality}: bad peak_allocated_bytes"
    assert all(len(v) == 2 for v in by_condition.values()), f"{modality}: expected 2 runs per condition"
    for rows in by_condition.values():
        assert sorted(row["repeat"] for row in rows) == [0, 1]
    return by_condition


def verify_fp16(data):
    assert data["status"] == "measured", f"fp16: status {data['status']}"
    assert data["model_dtype"] == "torch.float16"
    assert data["cache_bypassed"] is True
    assert data["model_load_in_timing"] is True
    assert len(data["runs"]) == 2, "fp16: expected 2 runs"
    assert sorted(row["repeat"] for row in data["runs"]) == [0, 1]
    assert len(data["feature_comparisons"]) == 2, "fp16: expected 2 comparisons"
    assert all(c["finite"] for c in data["feature_comparisons"])
    for run in data["runs"]:
        assert np.isfinite(run["wall_s"]) and run["wall_s"] > 0
        assert run["peak_allocated_bytes"] > 0
    return data["runs"]


def verify_accuracy(data):
    assert data["status"] in ("passed", "failed"), f"accuracy: status {data['status']}"
    assert data["gates"] == GATES, f"accuracy: gates {data['gates']} != {GATES}"
    assert data["n_movies"] == 4 and data["n_subjects"] == 1
    assert data["n_events"] == 60 and data["eval_lag"] == -5
    assert data["frames_per_clip"] == 64 and data["frequency_hz"] == 2
    assert data["clip_duration_s"] == 4 and data["cached_layers"] == 20
    assert data["original_alignment_windows_bitwise_equal"] == 40
    assert data["baseline_validation"]["n_timestamps"] == 16
    assert data["baseline_validation"]["all_passed"] is True
    assert len(data["runs"]) == 4, "accuracy: expected 4 runs"
    assert [r["run"] for r in data["runs"]] == ["MOVIE1_CC1", "MOVIE2_HO1", "MOVIE3_CC2", "MOVIE4_HO2"]
    for row in data["runs"]:
        for key in ("baseline_mean_parcel_r", "candidate_mean_parcel_r",
                    "mean_parcel_r_delta", "max_abs_parcel_r_delta", "relative_mse_change"):
            assert np.isfinite(row[key]), f"accuracy: non-finite {key}"
        passed = (abs(row["mean_parcel_r_delta"]) <= GATES["abs_mean_parcel_r_delta_le"]
                  and row["max_abs_parcel_r_delta"] <= GATES["max_abs_parcel_r_delta_le"]
                  and abs(row["relative_mse_change"]) <= GATES["abs_relative_mse_change_le"])
        assert row["passed"] == passed, "accuracy: stale passed flag"
    overall = all(row["passed"] for row in data["runs"])
    assert (data["status"] == "passed") == overall, "accuracy: status disagrees with rows"
    return data["runs"]


def verify_video_pair(original, half):
    assert original["model"] == "facebook/vjepa2-vitg-fpc64-256"
    assert half["model"] == original["model"]
    for key in ("selection", "torch", "hip", "transformers", "neuralset", "device"):
        assert original[key] == half[key], f"Video comparison differs in {key}"
    expected_shapes = [[20, 1408, 8]] * 4
    expected_timing = [{"frequency": 2.0, "duration": 4.0}] * 4
    for row in original["runs"] + half["runs"]:
        assert row["shapes"] == expected_shapes
        assert row["output_timing"] == expected_timing
        assert row["output_dtype"] == "float32"


def med(rows):
    return float(np.median([r["wall_s"] for r in rows]))


def maxpeak(rows):
    return max(r["peak_allocated_bytes"] for r in rows) / 2**30


def build_summary(fp32_video, fp16_runs, text, audio):
    fp32_base = fp32_video["baseline"]
    fp32_opt = fp32_video["optimized"]
    summary = []
    rows = [
        ("Video FP32", fp32_base, fp32_opt),
        ("Video FP16*", fp32_base, fp16_runs),
        ("Text BF16", text["baseline"], text["optimized"]),
        ("Audio FP32", audio["baseline"], audio["optimized"]),
    ]
    for name, base, cand in rows:
        orig = med(base)
        cand_med = med(cand)
        ratio = cand_med / orig
        cand_peak = maxpeak(cand)
        orig_peak = maxpeak(base)
        summary.append((name, orig, cand_med, ratio, orig_peak, cand_peak))
    return summary


def figure_performance(summary, fp32_video, fp16_runs, text, audio):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), layout="constrained")
    y = np.arange(4)
    sources = [(fp32_video["baseline"], fp32_video["optimized"]),
               (fp32_video["baseline"], fp16_runs),
               (text["baseline"], text["optimized"]),
               (audio["baseline"], audio["optimized"])]
    names = [name for name, *_ in summary]
    for i, (name, orig, cand, ratio, op, cp) in enumerate(summary):
        color, hatch = COLORS[i], HATCHES[i]
        axes[0].barh(y[i], ratio, color=color, hatch=hatch, height=.7)
        axes[1].barh(y[i]-.17, op, color="#BBBBBB", height=.3,
                     label="Original" if i == 0 else None)
        axes[1].barh(y[i]+.17, cp, color=color, hatch=hatch, height=.3,
                     label="Candidate" if i == 0 else None)
        base_rows, cand_rows = sources[i]
        cand_vals = [r["wall_s"] for r in cand_rows]
        axes[0].plot(np.array(cand_vals) / med(base_rows), [y[i]] * 2,
                     "o", color="black", markersize=3, zorder=5,
                     label="Candidate repetitions" if i == 0 else None)
    axes[0].axvline(1.0, color="black", ls=":", lw=.8, label="original = 1")
    axes[0].set_xlim(0, max(1.1, max(s[3] for s in summary) + .1))
    axes[0].set(xlabel="Normalized extraction time (original = 1)", title="A. Normalized extraction time")
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    for ax in axes:
        ax.set_yticks(y, names)
    axes[1].set(xlabel="Peak GPU tensor allocation (GiB)", title="B. Peak tensor allocation")
    axes[1].set_xlim(left=0)
    axes[1].legend(frameon=False, fontsize=7, loc="upper center",
                   bbox_to_anchor=(.5, -.20), ncol=2)
    for suffix in ("pdf", "png"):
        fig.savefig(REPORT / f"features-performance.{suffix}", dpi=200)
    plt.close(fig)


def verify_cohort(data):
    assert data["status"] in ("passed", "failed")
    assert data["gates"] == GATES and data["eval_lag"] == -5
    assert data["frames_per_clip"] == 64
    assert data["n_subjects"] == 168 and data["n_movies"] == 4
    assert data["n_comparisons"] == data["expected_comparisons"] == 672
    assert data["quantile_probabilities"] == [0,.05,.25,.5,.75,.95,1]
    assert [r["run"] for r in data["runs"]] == ["MOVIE1_CC1", "MOVIE2_HO1", "MOVIE3_CC2", "MOVIE4_HO2"]
    for row in data["runs"]:
        assert row["n_subjects"] == 168 and 0 <= row["n_failed_subjects"] <= 168
        quantiles = row["subject_mean_r_delta_quantiles"]
        assert len(quantiles) == 7 and np.isfinite(quantiles).all()
        assert np.all(np.diff(quantiles) >= 0)
        assert np.isclose(row["max_abs_subject_mean_r_delta"], max(abs(quantiles[0]), abs(quantiles[-1])))
        assert np.isclose(row["mean_parcel_r_delta"], row["candidate_mean_parcel_r"] - row["baseline_mean_parcel_r"], atol=1e-12)
        passed = (row["max_abs_subject_mean_r_delta"] <= GATES["abs_mean_parcel_r_delta_le"]
                  and row["max_abs_parcel_r_delta"] <= GATES["max_abs_parcel_r_delta_le"]
                  and row["max_abs_relative_mse_change"] <= GATES["abs_relative_mse_change_le"])
        assert (row["n_failed_subjects"] == 0) == passed
    failures = sum(row["n_failed_subjects"] for row in data["runs"])
    assert data["n_failed_comparisons"] == failures
    assert (data["status"] == "passed") == (failures == 0)
    return data["runs"]


def figure_accuracy(rows):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), layout="constrained")
    movies = np.arange(4)
    for i, row in enumerate(rows):
        low, p5, p25, median, p75, p95, high = row["subject_mean_r_delta_quantiles"]
        axes[0].plot([p5, p95], [i, i], color=COLORS[0], lw=1)
        axes[0].plot([p25, p75], [i, i], color=COLORS[0], lw=5)
        axes[0].plot(median, i, "D", color="black", ms=3)
        axes[0].plot([low, high], [i, i], "|", color=COLORS[0], ms=6)
    axes[0].axvline(0, color="gray", ls=":", lw=.8)
    axes[0].set_yticks(movies, [f"Movie {i+1}" for i in movies])
    axes[0].invert_yaxis()
    axes[0].set(title="A. Subject mean-correlation changes", xlabel="Candidate minus original Pearson r")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    metrics = [("max_abs_subject_mean_r_delta", 1e-4, "o", "Mean r"),
               ("max_abs_parcel_r_delta", 1e-3, "s", "Parcel r"),
               ("max_abs_relative_mse_change", 1e-3, "^", "Relative MSE")]
    for offset, (key, margin, marker, label), color in zip([-.18,0,.18], metrics, COLORS):
        axes[1].plot(movies + offset, [r[key] / margin for r in rows], marker,
                     color=color, ms=5, label=label)
    axes[1].axhline(1, color="black", ls="--", lw=.9, label="Gate")
    axes[1].set_xticks(movies, [str(i+1) for i in movies])
    axes[1].set(title="B. Worst change per metric", xlabel="Movie",
                ylabel="Maximum fraction of margin")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
                   bbox_to_anchor=(.5, -.20))
    for suffix in ("pdf", "png"):
        fig.savefig(REPORT / f"features-accuracy.{suffix}", dpi=200)
    plt.close(fig)


def main():
    REPORT.mkdir(exist_ok=True)
    video = load("features_video_float32")
    text = load("features_text")
    audio = load("features_audio")
    fp16 = load("features_video_float16")
    accuracy = load("video_precision_accuracy")
    cohort = load("video64_cohort_accuracy")

    verify_video_pair(video, fp16)
    video = verify_features(video, "video", 4, loading=True)
    text = verify_features(text, "text", 4, loading=False)
    audio = verify_features(audio, "audio", 4, loading=False)
    fp16_runs = verify_fp16(fp16)
    accuracy_rows = verify_accuracy(accuracy)
    cohort_rows = verify_cohort(cohort)

    summary = build_summary(video, fp16_runs, text, audio)
    figure_performance(summary, video, fp16_runs, text, audio)
    figure_accuracy(cohort_rows)

    table_rows = []
    for name, orig, cand, ratio, op, cp in summary:
        table_rows.append([name, f"{orig:.3f}", f"{cand:.3f}", f"{1/ratio:.2f}",
                           f"{op:.3f}", f"{cp:.3f}"])
    text_out = []
    text_out.append(r"This section reports the four feature-extraction workloads on the AMD RX~9070~XT. "
                    r"Video times include model loading; text and audio times exclude loading. "
                    r"The samples contain 32 video clips, 128 words, and four 60-second audio events. "
                    r"Each modality is a separate workload; these speed ratios do not measure a combined pipeline. "
                    r"Table~\ref{tab:features-performance} reports medians and ratios; "
                    r"Table~\ref{tab:features-raw} retains both repetitions. No confidence intervals are claimed.")
    text_out.append(table(
        "Median extraction time and peak GPU tensor allocation across two repetitions per condition. "
        "Speedup is original/candidate median; peak allocation is the maximum across repetitions. "
        "Video times include load; text and audio exclude load. "
        "*Video FP16 uses a separately recorded FP32 video baseline and is not interleaved with the FP16 runs.",
        "tab:features-performance",
        ["Condition", "Orig. (s)", "New (s)", "Speedup", "Orig. (GiB)", "New (GiB)"],
        table_rows))
    raw_rows = []
    for label, data in [("Video FP32", video), ("Text BF16", text), ("Audio FP32", audio)]:
        raw_rows.append([label,
                         ", ".join(f"{r['wall_s']:.3f}" for r in data["baseline"]),
                         ", ".join(f"{r['wall_s']:.3f}" for r in data["optimized"])])
    raw_rows.append(["Video FP16*",
                     ", ".join(f"{r['wall_s']:.3f}" for r in video["baseline"]),
                     ", ".join(f"{r['wall_s']:.3f}" for r in fp16_runs)])
    text_out.append(table(
        "Raw per-repetition extraction times (s). Video FP32 and FP16 are not interleaved; the FP16 "
        "candidate is compared against the separately recorded FP32 baseline median.",
        "tab:features-raw",
        ["Condition", "Original repetitions (s)", "Candidate repetitions (s)"],
        raw_rows))
    text_out.append(r"Figure~\ref{fig:features-performance} shows normalized extraction time with the original "
                    r"as 1 and the peak GPU tensor allocation. Candidate repetitions appear as points. "
                    r"FP16 changes feature values, so downstream score margins determine its acceptance.")
    text_out.append(r"Figure~\ref{fig:features-accuracy} reports score drift over all 168 existing subjects. "
                    r"The worst subject/movie pair determines acceptance; averaging subjects cannot hide a failure. "
                    r"The initial single-subject alignment check is retained in Table~\ref{tab:features-accuracy}.")
    acc_rows = []
    for i, row in enumerate(accuracy_rows):
        acc_rows.append([f"Movie {i + 1}",
                         f"{row['baseline_mean_parcel_r']:.6f}",
                         f"{row['candidate_mean_parcel_r']:.6f}",
                         scientific(row["mean_parcel_r_delta"]),
                         scientific(row["max_abs_parcel_r_delta"]),
                         scientific(row["relative_mse_change"])])
    text_out.append(table(
        "Existing-subject brain-score agreement for the FP16 video path. Baseline and candidate mean parcel "
        "correlations are shown to six decimals; deltas use scientific notation. "
        r"Margins: mean $|\Delta r| \leq 10^{-4}$, max parcel $|\Delta r| \leq 10^{-3}$, "
        r"$|\Delta \mathrm{MSE}|/\mathrm{MSE} \leq 10^{-3}$.",
        "tab:features-accuracy",
        ["Movie", "Orig. mean $r$", "New mean $r$", r"$\Delta$ mean $r$",
         r"Max $|\Delta r|$", r"Rel. MSE"],
        acc_rows))
    n_passed = sum(row["passed"] for row in accuracy_rows)
    text_out.append(f"{n_passed} of four movies satisfy all three prespecified margins. "
                    "These are engineering margins, not statistical equivalence tests.")
    if accuracy["status"] == "failed":
        text_out.append(r"\emph{The completed FP16 brain-accuracy experiment exceeded at least one "
                        r"prespecified margin; this is reported as a failure rather than discarded.}")
    text_out.append(table(
        "FP16 regression across all 168 existing subjects per movie. Worst changes take the maximum "
        "absolute change across subjects (and parcels for parcel r); each subject/movie pair uses the fixed margins.",
        "tab:cohort", ["Movie", "Orig. mean $r$", "New mean $r$", r"Max $|\Delta\bar r|$", r"Max $|\Delta r|$", r"Max rel. $|\Delta$MSE$|$"],
        [[str(i+1), f"{r['baseline_mean_parcel_r']:.6f}", f"{r['candidate_mean_parcel_r']:.6f}",
          scientific(r["max_abs_subject_mean_r_delta"]), scientific(r["max_abs_parcel_r_delta"]),
          scientific(r["max_abs_relative_mse_change"])] for i, r in enumerate(cohort_rows)]))
    failures = cohort["n_failed_comparisons"]
    text_out.append(f"Table~\\ref{{tab:cohort}} covers all 672 subject/movie pairs; {failures} exceed at least one margin. "
                    "The cohort is an existing cache, not an independently held-out population. "
                    "No participant identifiers or individual diagnostic rows are redistributed.")
    text_out += [
        r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{features-performance.pdf}"
        r"\caption{Extraction cost on the stated samples. Left: bars show candidate/original median time; "
        r"black points show the two candidate repetitions divided by the original median. "
        r"Right: gray and colored bars show original and candidate peak GPU tensor allocation, "
        r"respectively, taking the maximum of two runs. Video FP16 (*) was timed separately "
        r"from the original FP32 runs. Allocation excludes non-PyTorch VRAM.}"
        r"\label{fig:features-performance}\end{figure}",
        r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{features-accuracy.pdf}"
        r"\caption{Float16 video regression on four movies across 168 existing subjects. "
        r"Left: diamonds, subject median change in mean parcel r; thick lines, interquartile range; "
        r"thin lines, 5th--95th percentiles; ticks, extrema. These are subject distributions, not confidence intervals. "
        r"Right: the maximum for each metric divided by its fixed margin; these maxima may come from "
        r"different subjects. Circles show mean r, "
        r"squares maximum parcel r, and triangles relative labeled-vertex MSE. Values at or below 1 pass.}"
        r"\label{fig:features-accuracy}\end{figure}",
    ]
    (REPORT / "features-results.tex").write_text("\n\n".join(text_out) + "\n")


if __name__ == "__main__":
    main()
