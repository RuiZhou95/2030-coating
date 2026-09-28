#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research framework, Chinese validation figure and supplementary controls."""

from __future__ import annotations

import importlib.util
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import PROJECT_ROOT as PROJECT

RESULT_DIR = PROJECT / "results"
FIG_DIR = PROJECT / "figure"
SUPP_DIR = PROJECT / "supplementary"
SEED = 20260823

spec = importlib.util.spec_from_file_location("strict_modeling", PROJECT / "src/03_strict_modeling.py")
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def box(ax, x, y, w, h, text, face, edge="#334155", fontsize=18, weight="normal", text_color="#1F2937", linewidth=1.6):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=text_color)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#475569", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=color, shrinkA=3, shrinkB=3, connectionstyle=f"arc3,rad={rad}"))


def plot_framework():
    fig, ax = plt.subplots(figsize=(18.0, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#F8FAFC")
    box(ax, 0.04, 0.915, 0.92, 0.065, "INDEPENDENT DATA DOMAINS  |  SHARED PROTOCOL  |  NO LAYER COUPLING", "#17324D", edge="#17324D", fontsize=22, weight="bold", text_color="white", linewidth=0)

    # Domain-specific inputs and audits.
    box(ax, 0.035, 0.62, 0.205, 0.205, "PRIMER DOMAIN\n\nn = 72\nInhibitor - curing - interface\nTargets: salt spray, adhesion", "#D9EAF7", edge="#2F6690", fontsize=17, weight="bold")
    box(ax, 0.035, 0.25, 0.205, 0.205, "TOPCOAT DOMAIN\n\nn = 120\nBinder - ceramic - lubricant\nTargets: salt spray, wear", "#FBE4D5", edge="#D67B3D", fontsize=17, weight="bold")
    box(ax, 0.275, 0.645, 0.17, 0.155, "FACTORIAL AUDIT\n4 x 3 x 3 x 2 design\nFamily and level groups", "#EAF3F9", edge="#2F6690", fontsize=16, weight="bold")
    box(ax, 0.275, 0.275, 0.17, 0.155, "CLOSURE AUDIT\nEight parts sum to 100%\nZeros and collinearity", "#FDF0E7", edge="#D67B3D", fontsize=16, weight="bold")
    arrow(ax, 0.24, 0.722, 0.275, 0.722, color="#2F6690")
    arrow(ax, 0.24, 0.352, 0.275, 0.352, color="#D67B3D")

    # Shared protocol, applied independently to each domain.
    protocol = FancyBboxPatch((0.485, 0.15), 0.29, 0.70, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor="#FFFFFF", edgecolor="#64748B", linewidth=2.0)
    ax.add_patch(protocol)
    ax.text(0.63, 0.815, "SHARED ANALYTICAL PROTOCOL", ha="center", va="center", fontsize=19, fontweight="bold", color="#17324D")
    steps = [
        ("01", "Freeze holdout", "High-performance + controls"),
        ("02", "Encode design", "Raw / physical / ILR"),
        ("03", "Nested validation", "Random / family / level"),
        ("04", "Calibrate uncertainty", "Bootstrap + conformal"),
        ("05", "Stress-test signals", "Permutation + noise"),
        ("06", "Rank robust effects", "Closure-preserving shifts"),
    ]
    y_positions = [0.69, 0.58, 0.47, 0.36, 0.25, 0.14]
    for (num, title, subtitle), y in zip(steps, y_positions):
        box(ax, 0.515, y, 0.23, 0.085, f"{num}   {title}\n{subtitle}", "#EEF3F7", edge="#AAB7C4", fontsize=14, weight="bold", linewidth=1.0)
        if y > 0.14:
            arrow(ax, 0.63, y, 0.63, y - 0.025, color="#94A3B8")
    arrow(ax, 0.445, 0.722, 0.485, 0.68, color="#2F6690", rad=-0.12)
    arrow(ax, 0.445, 0.352, 0.485, 0.32, color="#D67B3D", rad=0.12)

    # Layer-specific outputs.
    box(ax, 0.815, 0.62, 0.15, 0.205, "PRIMER OUTPUT\n\nConservative Pareto\nSalt spray up\nAdhesion up", "#D9EAF7", edge="#2F6690", fontsize=17, weight="bold")
    box(ax, 0.815, 0.25, 0.15, 0.205, "TOPCOAT OUTPUT\n\nConservative Pareto\nSalt spray up\nWear loss down", "#FBE4D5", edge="#D67B3D", fontsize=17, weight="bold")
    arrow(ax, 0.775, 0.68, 0.815, 0.72, color="#2F6690", rad=-0.10)
    arrow(ax, 0.775, 0.32, 0.815, 0.35, color="#D67B3D", rad=0.10)

    box(ax, 0.16, 0.035, 0.68, 0.07, "Frozen samples are retrospective holdouts - shortlisted formulations are not new experimental validation", "#FFF7E8", edge="#E2A23A", fontsize=16, weight="bold")
    fig.savefig(FIG_DIR / "fig1_research_framework.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_research_framework.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_validation_chinese():
    agg = pd.read_csv(RESULT_DIR / "nested_cv_summary.csv", encoding="utf-8-sig")
    selected = pd.read_csv(RESULT_DIR / "final_model_selection.csv", encoding="utf-8-sig")
    names = {
        ("primer", "adhesion_mpa_or_grade"): "Primer\nadhesion",
        ("primer", "salt_spray_pass_h"): "Primer\nsalt spray",
        ("primer", "wear_mass_loss_mg"): "Primer\nwear (aux.)",
        ("topcoat", "salt_spray_pass_h"): "Topcoat\nsalt spray",
        ("topcoat", "wear_mass_loss_mg"): "Topcoat\nwear",
    }
    schemes = [("random_repeated", "Repeated random split"), ("family_logo", "Formulation-category holdout"), ("level_logo", "Key-composition-level holdout")]
    labels, rows = [], []
    for _, s in selected.iterrows():
        labels.append(names[(s["layer"], s["target"])])
        q = agg[(agg["layer"] == s["layer"]) & (agg["target"] == s["target"]) & (agg["representation"] == s["representation"]) & (agg["model"] == s["model"])]
        rows.append({r["scheme"]: r["pooled_r2"] for _, r in q.iterrows()})
    matrix = np.array([[row.get(s[0], np.nan) for s in schemes] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(16.0, 7.5), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.24
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for j, ((_, legend_label), color) in enumerate(zip(schemes, colors)):
        ax.bar(x + (j - 1) * width, matrix[:, j], width=width, color=color, label=legend_label)
    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Pooled outer-test R²")
    ax.set_ylim(min(-0.50, float(np.nanmin(matrix) - 0.08)), 1.07)
    ax.set_title("Interpolation under random splits versus structured holdout", loc="left", fontweight="normal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.57, 1.15), ncol=3, frameon=False)
    ax.grid(axis="y", color="#D5D8DC", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / "fig3_validation_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_validation_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_validation_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def zero_replacement_sensitivity():
    rows = []
    for target in ["salt_spray_pass_h", "wear_mass_loss_mg"]:
        train, _ = sm.prepare_target("topcoat", target)
        y = train[target].to_numpy(float)
        schemes = sm.outer_schemes("topcoat", train)
        for factor in [0.1, 0.5, 0.9]:
            for scheme_name in ["family_logo", "level_logo"]:
                true, pred = [], []
                for fold_id, (tr, te, inner_groups) in enumerate(schemes[scheme_name]):
                    pipe = Pipeline([("repr", sm.TopcoatRepresentation("ilr", zero_factor=factor)), ("scale", StandardScaler()), ("model", Ridge())])
                    search = sm.fit_search(pipe, {"model__alpha": [0.01, 0.1, 1, 10, 100]}, train.iloc[tr], y[tr], scheme_name, inner_groups, fold_id)
                    true.extend(y[te])
                    pred.extend(np.asarray(search.predict(train.iloc[te])).ravel())
                m = sm.metrics(true, pred)
                rows.append({"target": target, "zero_factor": factor, "scheme": scheme_name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "zero_replacement_sensitivity.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)
    for ax, target, title in zip(axes, ["salt_spray_pass_h", "wear_mass_loss_mg"], ["Topcoat salt-spray lifetime", "Topcoat wear loss"]):
        g = df[df["target"] == target]
        for scheme, label, color in [("family_logo", "Formulation-category holdout", "#F58518"), ("level_logo", "Key-composition-level holdout", "#54A24B")]:
            q = g[g["scheme"] == scheme]
            ax.plot(q["zero_factor"], q["r2"], marker="o", markersize=10, linewidth=2.5, color=color, label=label)
        ax.set_xlabel("Zero replacement factor x minimum positive part")
        ax.set_ylabel("Pooled R² of ILR-Ridge")
        ax.set_xticks([0.1, 0.5, 0.9], ["0.10", "0.50", "0.90"])
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    fig.savefig(SUPP_DIR / "figS2_zero_replacement_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(SUPP_DIR / "figS2_zero_replacement_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(SUPP_DIR / "figS2_zero_replacement_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)


def plot_negative_controls():
    perm = pd.read_csv(RESULT_DIR / "label_permutation_controls.csv", encoding="utf-8-sig")
    noise = pd.read_csv(RESULT_DIR / "random_noise_controls.csv", encoding="utf-8-sig")
    keys = list(noise[["layer", "target"]].itertuples(index=False, name=None))
    label_map = {
        ("primer", "adhesion_mpa_or_grade"): "Primer\nadhesion",
        ("primer", "salt_spray_pass_h"): "Primer\nsalt spray",
        ("primer", "wear_mass_loss_mg"): "Primer\nwear (aux.)",
        ("topcoat", "salt_spray_pass_h"): "Topcoat\nsalt spray",
        ("topcoat", "wear_mass_loss_mg"): "Topcoat\nwear",
    }
    labels = [label_map[(a, b)] for a, b in keys]
    data = [perm[(perm["layer"] == a) & (perm["target"] == b)]["r2"].dropna() for a, b in keys]
    fig, ax = plt.subplots(figsize=(15.0, 6.8), constrained_layout=True)
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor="#DCE6F1", edgecolor="#4C78A8")
    observed = [noise[(noise["layer"] == a) & (noise["target"] == b)]["observed_r2"].iloc[0] for a, b in keys]
    ax.scatter(np.arange(1, len(keys) + 1), observed, color="#C62828", s=95, zorder=4, label="Observed structured-holdout R²")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(np.arange(1, len(keys) + 1), labels)
    ax.set_ylabel("Structured-holdout R²")
    ax.set_title("Label-permutation negative control (100 repeats)", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(SUPP_DIR / "figS1_label_permutation.png", dpi=600, bbox_inches="tight")
    fig.savefig(SUPP_DIR / "figS1_label_permutation.pdf", bbox_inches="tight")
    fig.savefig(SUPP_DIR / "figS1_label_permutation.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    style()
    plot_framework()
    plot_validation_chinese()
    zero_replacement_sensitivity()
    plot_negative_controls()
    shutil.copy2(FIG_DIR / "fig1_data_space.png", FIG_DIR / "fig2_data_space.png")
    shutil.copy2(FIG_DIR / "fig1_data_space.pdf", FIG_DIR / "fig2_data_space.pdf")
    shutil.copy2(FIG_DIR / "fig1_data_space.svg", FIG_DIR / "fig2_data_space.svg")
    print("FRAMEWORK_SUPPLEMENTARY_OK")
    print(pd.read_csv(RESULT_DIR / "zero_replacement_sensitivity.csv", encoding="utf-8-sig").to_string(index=False))


if __name__ == "__main__":
    main()
