"""Regenerate report tables and figures from the recorded measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
CASES = [("rocm_hcp_b1", "ROCm, batch 1"), ("rocm_hcp_b8", "ROCm, batch 8"),
         ("cpu26_hcp_b1", "CPU Torch 2.6, batch 1")]
VARIANTS = [("baseline", "Baseline"), ("optimized", "Fused readout"), ("sdpa", "Fused + SDPA")]


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def table(caption, label, headings, rows):
    return "\n".join([
        r"\begin{table}[htbp]\centering\sffamily\small",
        rf"\caption{{{caption}}}\label{{{label}}}",
        r"\begin{tabular}{@{}l" + "r" * (len(headings) - 1) + r"@{}}\toprule",
        " & ".join(headings) + r" \\\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule\end{tabular}\end{table}",
    ])


def main():
    data = {stem: json.loads((ROOT / "results" / f"{stem}.json").read_text()) for stem, _ in CASES}
    brain = json.loads((ROOT / "results/hcp_regression.json").read_text())
    assert brain["status"] == "passed"
    expected = {
        "rocm_hcp_b1": ("cuda:0", 1, 40, 80, 10),
        "rocm_hcp_b8": ("cuda:0", 8, 5, 60, 10),
        "cpu26_hcp_b1": ("cpu", 1, 40, 40, 5),
    }
    comparisons = ["pooling_only", "optimized", "sdpa_only", "sdpa"]
    for stem, result in data.items():
        device, batch_size, n_batches, steps, warmup = expected[stem]
        assert result["input"]["kind"] == "cached_feature_batches"
        assert result["input"]["n_batches"] == n_batches
        assert result["input"]["batch_sizes"] == [batch_size] * n_batches
        assert result["environment"]["device"] == device
        assert result["environment"]["torch"].startswith("2.6.0" if device == "cpu" else "2.12.0+rocm")
        assert result["timing"]["steps"] == steps and result["timing"]["warmup"] == warmup
        for name in comparisons:
            assert result["correctness"][f"{name}_vs_baseline"]["allclose"] is True
        for name, _ in VARIANTS:
            values = result["timing"][name]
            assert np.isfinite([values["median_ms"], values["p25_ms"], values["p75_ms"]]).all()
    REPORT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    colors, hatches = ["#0072B2", "#D55E00", "#009E73"], ["", "//", ".."]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), layout="constrained")
    for index, (name, label) in enumerate(VARIANTS):
        timings = [data[stem]["timing"][name] for stem, _ in CASES[:2]]
        median = np.array([x["median_ms"] for x in timings])
        quartiles = np.array([[x["p25_ms"] for x in timings], [x["p75_ms"] for x in timings]])
        position = np.arange(2) + (index - 1) * .24
        axes[0].bar(position, median, .24, color=colors[index], hatch=hatches[index], label=label,
                    yerr=np.stack([median-quartiles[0], quartiles[1]-median]), capsize=2)
        axes[1].bar(position, [x["memory"]["cuda"]["peak_allocated_bytes"] / 2**20 for x in timings],
                    .24, color=colors[index], hatch=hatches[index])
    for axis in axes:
        axis.set_xticks([0, 1], ["Batch 1", "Batch 8"])
        axis.set_ylim(bottom=0)
    axes[0].set(title="A. Forward latency", ylabel="Median latency (ms; IQR bars)")
    axes[1].set(title="B. Peak tensor allocation", ylabel="PyTorch GPU allocation (MiB)")
    axes[0].legend(frameon=False, fontsize=7)
    for suffix in ["pdf", "png"]:
        fig.savefig(REPORT / f"performance.{suffix}", dpi=200)
    plt.close(fig)

    rows = brain["runs"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), layout="constrained")
    axes[0].semilogy(x, [r["pooled_max_abs"] for r in rows], "o-", color=colors[0], label="Pooled")
    axes[0].semilogy(x, [r["unpooled_max_abs"] for r in rows], "s--", color=colors[1], label="Unpooled")
    axes[0].axhline(1e-5, color="black", ls=":", label="Absolute tolerance")
    axes[0].set(title="A. Prediction agreement", ylabel="Maximum absolute difference")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].semilogy(x, [r["max_abs_per_parcel_r_delta"] for r in rows], "o-", color=colors[0])
    axes[1].axhline(1e-6, color="black", ls=":", label="Numerical score gate")
    axes[1].set(title="B. Brain-score agreement", ylabel=r"Maximum parcel $|\Delta r|$")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.set_xticks(x, [f"Movie {i+1}" for i in x])
    for suffix in ["pdf", "png"]:
        fig.savefig(REPORT / f"fidelity.{suffix}", dpi=200)
    plt.close(fig)

    gpu = data["rocm_hcp_b8"]
    model = gpu["model"]
    reduction = 100 * (1 - model["optimized_parameter_count"] / model["parameter_count"])
    summary = {
        "ParameterReduction": f"{reduction:.2f}",
        "ReadoutSpeedup": f"{gpu['timing']['baseline']['median_ms']/gpu['timing']['optimized']['median_ms']:.2f}",
        "MaximumParcelError": scientific(max(r["max_abs_per_parcel_r_delta"] for r in rows)),
    }
    (REPORT / "summary.tex").write_text("\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in summary.items()) + "\n")
    text = [rf"The encoder parameter count decreases from {model['parameter_count']:,} to "
            rf"{model['optimized_parameter_count']:,} ({reduction:.2f}\%). "
            rf"Float32 parameter storage decreases from {model['parameter_bytes']/2**20:.2f} to "
            rf"{model['optimized_parameter_bytes']/2**20:.2f} MiB. "
            r"Table~\ref{tab:runtime} reports forward latency; each runtime is a separate comparison."]
    runtime_rows = []
    for stem, label in CASES:
        values = data[stem]["timing"]
        runtime_rows.append([label, *[f"{values[k]['median_ms']:.2f}" for k, _ in VARIANTS],
                             f"{values['baseline']['median_ms']/values['optimized']['median_ms']:.2f}"])
    text.append(table("Median forward latency (ms) and baseline/fused speed ratio", "tab:runtime",
                      ["Runtime", "Baseline", "Fused", "Fused + SDPA", "Ratio"], runtime_rows))
    ablations = [("baseline", "Baseline"), ("pooling_only", "Pooling only"), ("optimized", "Fused readout"),
                 ("sdpa_only", "SDPA only"), ("sdpa", "Fused + SDPA")]
    ablation_rows = []
    for name, label in ablations:
        value = gpu["timing"].get(name)
        if value is not None:
            ablation_rows.append([label, f"{value['median_ms']:.2f}",
                                 f"[{value['p25_ms']:.2f}, {value['p75_ms']:.2f}]",
                                 f"{value['memory']['cuda']['peak_allocated_bytes']/2**20:.2f}"])
    text += [r"Table~\ref{tab:ablation} separates the candidate changes on ROCm at batch 8. "
             r"The intervals are sample quartiles, not confidence intervals. Allocator peaks include input tensors; they are not physical VRAM or activation-only measurements.",
             table("ROCm batch-8 ablation and peak tensor allocation", "tab:ablation",
                   ["Variant", "Median (ms)", "IQR (ms)", "Peak (MiB)"], ablation_rows)]
    conversion = gpu["model"]["conversion_memory"]
    saved_ms = gpu["timing"]["baseline"]["median_ms"] - gpu["timing"]["optimized"]["median_ms"]
    text.append(
        f"In-place batch-8 conversion takes {conversion['time_ms']:.2f} ms. "
        f"Its sampled incremental process RSS is {conversion['rss']['incremental_rss_bytes']/2**20:.2f} MiB "
        f"and its peak GPU tensor allocation is {conversion['cuda']['peak_allocated_bytes']/2**20:.2f} MiB. "
        f"Dividing this conversion time by the observed per-call saving gives about {conversion['time_ms']/saved_ms:.0f} batch-8 calls to amortize conversion. "
        r"This is an in-place estimate, not a cold-loader measurement. A single four-movie pass has only five batch-8 calls, so a total-runtime benefit is not established for that workload. "
        r"Parameter savings apply immediately; forward-time savings favor reusing the converted model."
    )
    text.append(r"Table~\ref{tab:fidelity} reports the existing-subject regression check. All four movies pass pooled and unpooled numerical gates.")
    text.append(table("Mean parcel correlation and numerical differences for one existing evaluation subject", "tab:fidelity",
                      ["Movie", "Baseline $r$", "Fused $r$", r"$\Delta$ mean $r$", r"Max parcel $|\Delta r|$"],
                      [[str(i+1), f"{r['baseline_mean_parcel_r']:.6f}", f"{r['optimized_mean_parcel_r']:.6f}",
                        scientific(r["mean_parcel_r_delta"]), scientific(r["max_abs_per_parcel_r_delta"])] for i,r in enumerate(rows)]))
    text.append("The largest pooled and unpooled absolute prediction differences are "
                + scientific(max(r["pooled_max_abs"] for r in rows)) + " and "
                + scientific(max(r["unpooled_max_abs"] for r in rows)) + ", respectively. The largest absolute relative change in labeled-vertex MSE is "
                + scientific(max(abs(r["relative_mse_change"]) for r in rows)) + ". No population-level equivalence test is claimed.")
    for batch in [1, 8]:
        compiled = json.loads((ROOT / "results" / f"compile_hcp_b{batch}.json").read_text())
        assert compiled["failed_elements"] == 0
        eager, fast = compiled["timing"]["eager"], compiled["timing"]["compiled"]
        text.append(f"In the separate batch-{batch} compiler experiment, fused eager inference takes {eager['median_ms']:.2f} ms and compiled inference {fast['median_ms']:.2f} ms "
                    f"(observed timing ratio {eager['median_ms']/fast['median_ms']:.2f}). The first compiled call takes {fast['first_call_ms']/1000:.2f} s with the existing compiler cache; this is not a clean-cache compilation estimate.")
    reverse_path = ROOT / "results/rocm_hcp_b8_reverse.json"
    if reverse_path.exists():
        reverse_result = json.loads(reverse_path.read_text())
        assert reverse_result["conditions"]["reverse_order"] is True
        assert reverse_result["input"]["kind"] == "cached_feature_batches"
        assert reverse_result["input"]["batch_sizes"] == [8] * 5
        assert reverse_result["environment"] == gpu["environment"]
        assert reverse_result["timing"]["steps"] == gpu["timing"]["steps"]
        assert reverse_result["timing"]["warmup"] == gpu["timing"]["warmup"]
        for name in comparisons:
            assert reverse_result["correctness"][f"{name}_vs_baseline"]["allclose"] is True
        reverse = reverse_result["timing"]
        base, fused = reverse["baseline"]["median_ms"], reverse["optimized"]["median_ms"]
        text.append(f"With the condition order reversed at batch 8, baseline and fused medians are {base:.2f} and {fused:.2f} ms, respectively (ratio {base/fused:.2f}).")
    (REPORT / "results.tex").write_text("\n\n".join(text) + "\n")


if __name__ == "__main__":
    main()
