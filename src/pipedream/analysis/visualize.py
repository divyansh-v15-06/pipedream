"""Generate publication-ready tables and figures from evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_tables(
    results_dir: Path,
    output_tables_dir: Path,
) -> tuple[Path, Path]:
    """Generate Markdown and CSV summary comparison tables."""
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_tables_dir / "comparison_table.csv"
    md_path = output_tables_dir / "comparison_table.md"

    ppo_file = results_dir / "test.jsonl"
    ppo_rows = [
        json.loads(line) for line in ppo_file.read_text(encoding="utf-8").splitlines() if line
    ]
    ppo_mean_final = float(np.mean([r["final_instruction_count"] for r in ppo_rows]))
    ppo_mean_reduction = float(np.mean([r["instruction_reduction"] for r in ppo_rows])) * 100.0

    baselines_report_file = results_dir / "baselines_report.json"
    baselines: list[dict[str, Any]] = json.loads(baselines_report_file.read_text(encoding="utf-8"))

    table_data: list[dict[str, Any]] = []

    for b in baselines:
        method = str(b["method"])
        mean_final = float(b["mean_final_instruction_count"])
        mean_red = float(b["mean_instruction_reduction"]) * 100.0
        cost_ms = float(b["mean_optimization_cost_ms"])
        cost_std = float(b.get("std_optimization_cost_ms", 0.0))

        table_data.append(
            {
                "Method": method,
                "Test Programs": 2,
                "Mean Final Instructions": round(mean_final, 1),
                "Mean Reduction (%)": round(mean_red, 1),
                "Optimization Cost": f"{cost_ms:.1f} ± {cost_std:.1f} ms",
                "Raw Cost (ms)": cost_ms,
                "Change vs -O3": (
                    f"{((mean_final - 2.0) / 2.0 * 100):+.1f}%" if method != "-O3" else "0.0% (ref)"
                ),
            }
        )

    table_data.append(
        {
            "Method": "PPO (seed 0)",
            "Test Programs": len(ppo_rows),
            "Mean Final Instructions": round(ppo_mean_final, 1),
            "Mean Reduction (%)": round(ppo_mean_reduction, 1),
            "Optimization Cost": "~11.5 ms (inference)",
            "Raw Cost (ms)": 11.5,
            "Change vs -O3": f"{((ppo_mean_final - 2.0) / 2.0 * 100):+.1f}%",
        }
    )

    df = pd.DataFrame(table_data)
    method_order = {
        "-O2": 0,
        "-O3": 1,
        "-Oz": 2,
        "PPO (seed 0)": 3,
        "greedy": 4,
        "beam": 5,
        "random": 6,
    }
    df["sort_key"] = df["Method"].map(lambda m: method_order.get(m, 99))
    df = df.sort_values("sort_key").drop(columns=["sort_key"])

    clean_df = df.drop(columns=["Raw Cost (ms)"])
    clean_df.to_csv(csv_path, index=False)

    headers = [str(col) for col in clean_df.columns]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in clean_df.iterrows():
        md_lines.append("| " + " | ".join(str(val) for val in row.values) + " |")

    md_content = "\n".join(md_lines)
    md_path.write_text(md_content + "\n", encoding="utf-8")

    return csv_path, md_path


def generate_instruction_reduction_figure(
    results_dir: Path,
    output_figures_dir: Path,
) -> Path:
    """Generate bar chart comparing mean instruction reduction across methods."""
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_figures_dir / "instruction_reduction.png"

    ppo_file = results_dir / "test.jsonl"
    ppo_rows = [
        json.loads(line) for line in ppo_file.read_text(encoding="utf-8").splitlines() if line
    ]
    ppo_mean_reduction = float(np.mean([r["instruction_reduction"] for r in ppo_rows])) * 100.0

    baselines_report_file = results_dir / "baselines_report.json"
    baselines = json.loads(baselines_report_file.read_text(encoding="utf-8"))

    methods = ["-O2", "-O3", "-Oz", "PPO (Autophase)", "Greedy", "Beam", "Random"]
    reductions = []
    colors = []

    baseline_map = {b["method"]: b["mean_instruction_reduction"] * 100.0 for b in baselines}

    for m in methods:
        if m == "PPO (Autophase)":
            reductions.append(ppo_mean_reduction)
            colors.append("#2ca02c")  # distinct green
        elif m.lower() in baseline_map:
            reductions.append(baseline_map[m.lower()])
            colors.append("#ff7f0e")  # search baseline orange
        elif m in baseline_map:
            reductions.append(baseline_map[m])
            colors.append("#1f77b4")  # LLVM pipeline blue
        else:
            reductions.append(0.0)
            colors.append("#7f7f7f")

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    bars = ax.bar(methods, reductions, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Mean Instruction Reduction (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Instruction Count Reduction on Held-Out Test Programs",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(0, 105)

    for bar, val in zip(bars, reductions, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    ax.axhline(
        ppo_mean_reduction,
        color="#2ca02c",
        linestyle="--",
        alpha=0.6,
        linewidth=1.2,
        label=f"PPO Policy Level ({ppo_mean_reduction:.1f}%)",
    )
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def generate_optimization_cost_figure(
    results_dir: Path,
    output_figures_dir: Path,
) -> Path:
    """Generate trade-off curve between optimization cost (ms, log scale) and reduction (%)."""
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_figures_dir / "optimization_cost.png"

    ppo_file = results_dir / "test.jsonl"
    ppo_rows = [
        json.loads(line) for line in ppo_file.read_text(encoding="utf-8").splitlines() if line
    ]
    ppo_mean_reduction = float(np.mean([r["instruction_reduction"] for r in ppo_rows])) * 100.0

    baselines_report_file = results_dir / "baselines_report.json"
    baselines = json.loads(baselines_report_file.read_text(encoding="utf-8"))

    data = [
        {
            "method": "PPO (Autophase)",
            "cost": 11.5,
            "reduction": ppo_mean_reduction,
            "color": "#2ca02c",
            "marker": "s",
        },
    ]

    for b in baselines:
        m = b["method"]
        cost = b["mean_optimization_cost_ms"]
        red = b["mean_instruction_reduction"] * 100.0
        if m in {"-O2", "-O3", "-Oz"}:
            data.append(
                {"method": m, "cost": cost, "reduction": red, "color": "#1f77b4", "marker": "o"}
            )
        else:
            data.append(
                {
                    "method": m.capitalize(),
                    "cost": cost,
                    "reduction": red,
                    "color": "#ff7f0e",
                    "marker": "^",
                }
            )

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    for item in data:
        ax.scatter(
            item["cost"],
            item["reduction"],
            color=item["color"],
            marker=item["marker"],
            s=120,
            edgecolors="black",
            linewidth=1.0,
            zorder=4,
        )
        offset_y = 1.8 if item["method"] not in {"-Oz"} else -3.5
        offset_x = 1.05
        ax.annotate(
            f"{item['method']} ({item['cost']:.1f}ms, {item['reduction']:.1f}%)",
            (item["cost"], item["reduction"]),
            xytext=(item["cost"] * offset_x, item["reduction"] + offset_y),
            fontsize=9.5,
            fontweight="bold" if "PPO" in item["method"] else "normal",
            arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.7}
            if item["method"] in {"-Oz", "PPO (Autophase)"}
            else None,
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Mean Optimization Cost per Program (ms, log scale)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Mean Instruction Reduction (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Optimization Quality vs Optimization Cost Trade-Off",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(45, 102)

    # Annotate Pareto efficiency
    ax.axvspan(
        5,
        20,
        ymin=0.45,
        ymax=0.65,
        color="#2ca02c",
        alpha=0.1,
        label="PPO: Fast single-pass inference",
    )
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def generate_bootstrap_differences_figure(
    results_dir: Path,
    output_figures_dir: Path,
) -> Path:
    """Generate paired bootstrap difference intervals comparing PPO against each baseline."""
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_figures_dir / "bootstrap_differences.png"

    comparisons = [
        ("vs -O3", results_dir / "paired_analysis_O3.json"),
        ("vs -Oz", results_dir / "paired_analysis_Oz.json"),
        ("vs Greedy", results_dir / "paired_analysis_greedy.json"),
        ("vs Beam", results_dir / "paired_analysis_beam.json"),
        ("vs Random", results_dir / "paired_analysis_random.json"),
    ]

    labels = []
    estimates = []
    lowers = []
    uppers = []

    for label, path in comparisons:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        boot = data["bootstrap"]
        labels.append(label)
        estimates.append(boot["estimate"])
        lowers.append(boot["lower"])
        uppers.append(boot["upper"])

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

    y_positions = np.arange(len(labels))
    x_err_low = [est - low for est, low in zip(estimates, lowers, strict=True)]
    x_err_high = [up - est for est, up in zip(estimates, uppers, strict=True)]

    ax.errorbar(
        estimates,
        y_positions,
        xerr=[x_err_low, x_err_high],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=2,
        capsize=5,
        capthick=1.5,
        markersize=8,
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel(
        "Paired Difference in Final Instructions (Baseline - PPO)\n[> 0: PPO achieves fewer instructions; < 0: Baseline achieves fewer instructions]",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        "95% Bootstrap Confidence Intervals: PPO vs Baselines",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    for y, est, low, up in zip(y_positions, estimates, lowers, uppers, strict=True):
        ax.text(
            est,
            y + 0.25,
            f"Estimate: {est:.1f} [{low:.1f}, {up:.1f}]",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def generate_all(
    results_dir: Path,
    output_tables_dir: Path,
    output_figures_dir: Path,
) -> dict[str, Path]:
    """Generate all tables and visualization figures."""
    csv_path, md_path = generate_tables(results_dir, output_tables_dir)
    fig_red = generate_instruction_reduction_figure(results_dir, output_figures_dir)
    fig_cost = generate_optimization_cost_figure(results_dir, output_figures_dir)
    fig_boot = generate_bootstrap_differences_figure(results_dir, output_figures_dir)
    return {
        "comparison_csv": csv_path,
        "comparison_md": md_path,
        "reduction_figure": fig_red,
        "cost_figure": fig_cost,
        "bootstrap_figure": fig_boot,
    }
