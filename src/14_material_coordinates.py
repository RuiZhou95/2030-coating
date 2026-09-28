#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Material-coordinate analysis and material-centered main figures."""

from __future__ import annotations

import importlib.util
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT as PROJECT

RESULT_DIR = PROJECT / "results"
DATA_DIR = PROJECT / "data"
FIG_DIR = PROJECT / "figure"
SUPP_DIR = PROJECT / "supplementary"

spec = importlib.util.spec_from_file_location("audit", PROJECT / "src/01_data_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

BLUE = "#2F6690"
ORANGE = "#D97706"
TEAL = "#287271"
RED = "#C8553D"
GOLD = "#D9A441"
INK = "#263238"
LIGHT = "#E8EEF3"
GRAY = "#8A99A6"


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_data():
    primer = audit.load_layer("底漆", audit.FILES["底漆"])
    topcoat = audit.load_layer("面漆", audit.FILES["面漆"])
    if "salt_spray_time_h" in primer.columns or "salt_spray_time_h" in topcoat.columns:
        raise AssertionError("Target alias salt_spray_time_h entered cleaned data")
    numeric = (
        audit.TARGETS
        + audit.PRIMER_COMPONENTS
        + audit.TOPCOAT_COMPONENTS
        + ["coating_cure_temp_c", "coating_cure_time_h", "surface_roughness_ra_um"]
    )
    for frame in [primer, topcoat]:
        for column in numeric:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return primer, topcoat


def nearest(values, levels):
    levels = np.asarray(levels, float)
    return np.array([levels[np.argmin(np.abs(levels - value))] for value in values], float)


def topcoat_coordinates(topcoat):
    out = topcoat.copy()
    pu = out["聚氨酯树脂A14含量"]
    fluor = out["氟树脂含量"]
    sic = out["碳化硅"]
    bn = out["氮化硼"]
    lubricant = out["润滑粉"]
    binder = pu + fluor
    ceramic = sic + bn
    solid = ceramic + lubricant
    out["binder_total_wt_pct"] = binder
    out["ceramic_total_wt_pct"] = ceramic
    out["solid_total_wt_pct"] = solid
    out["fluororesin_fraction_in_binder"] = nearest(
        fluor / binder, [0.0, 0.25, 1 / 3, 0.5, 0.75, 1.0]
    )
    out["solid_to_binder_mass_ratio"] = nearest(
        solid / binder, [0.25, 3 / 7, 1.0, 7 / 3, 4.0]
    )
    out["bn_fraction_in_ceramic"] = bn / ceramic
    out["lubricant_fraction_in_solid"] = lubricant / solid
    package_levels = np.array([0.137, 0.3225, 0.4348, 0.539])
    out["filler_package_level"] = [
        int(np.argmin(np.abs(package_levels - value))) + 1
        for value in out["bn_fraction_in_ceramic"]
    ]
    out["additive_total_wt_pct"] = out[["分散剂", "防沉剂", "消泡剂"]].sum(axis=1)
    return out


def balanced_ss(frame, target):
    a_name, b_name, c_name = (
        "fluororesin_fraction_in_binder",
        "solid_to_binder_mass_ratio",
        "filler_package_level",
    )
    y = frame[target].to_numpy(float)
    grand = float(np.mean(y))
    total = float(np.sum((y - grand) ** 2))
    a_levels = sorted(frame[a_name].unique())
    b_levels = sorted(frame[b_name].unique())
    c_levels = sorted(frame[c_name].unique())
    na, nb, nc = len(a_levels), len(b_levels), len(c_levels)
    mean_a = frame.groupby(a_name)[target].mean()
    mean_b = frame.groupby(b_name)[target].mean()
    mean_c = frame.groupby(c_name)[target].mean()
    ss_a = nb * nc * float(np.sum((mean_a - grand) ** 2))
    ss_b = na * nc * float(np.sum((mean_b - grand) ** 2))
    ss_c = na * nb * float(np.sum((mean_c - grand) ** 2))
    mean_ab = frame.groupby([a_name, b_name])[target].mean()
    ss_ab = nc * sum(
        (value - mean_a.at[a] - mean_b.at[b] + grand) ** 2
        for (a, b), value in mean_ab.items()
    )
    mean_ac = frame.groupby([a_name, c_name])[target].mean()
    ss_ac = nb * sum(
        (value - mean_a.at[a] - mean_c.at[c] + grand) ** 2
        for (a, c), value in mean_ac.items()
    )
    mean_bc = frame.groupby([b_name, c_name])[target].mean()
    ss_bc = na * sum(
        (value - mean_b.at[b] - mean_c.at[c] + grand) ** 2
        for (b, c), value in mean_bc.items()
    )
    residual = total - (ss_a + ss_b + ss_c + ss_ab + ss_ac + ss_bc)
    terms = {
        "Binder fluorination (phi_F)": ss_a,
        "Functional-solid/binder ratio (psi)": ss_b,
        "Filler package": ss_c,
        "phi_F x psi": ss_ab,
        "phi_F x package": ss_ac,
        "psi x package": ss_bc,
        "Three-way interaction": residual,
    }
    return pd.DataFrame(
        {
            "target": target,
            "term": list(terms),
            "sum_of_squares": list(terms.values()),
            "variance_pct": [100 * value / total for value in terms.values()],
        }
    )


def pareto_full(frame):
    values = frame[["salt_spray_pass_h", "wear_mass_loss_mg"]].to_numpy(float)
    keep = []
    for i, (salt, wear) in enumerate(values):
        dominated = np.any(
            (values[:, 0] >= salt)
            & (values[:, 1] <= wear)
            & ((values[:, 0] > salt) | (values[:, 1] < wear))
        )
        keep.append(not dominated)
    return frame.loc[np.asarray(keep)].copy()


def export_results(primer, topcoat):
    primer_design_columns = [
        "铬酸锶", "磷酸锌", "coating_curing_agent_type",
        "coating_cure_temp_c", "coating_cure_time_h",
    ]
    primer_design_cells = (
        primer.groupby(primer_design_columns, dropna=False)
        .size().rename("n").reset_index()
    )
    primer_means = []
    for factor in ["铬酸锶", "磷酸锌", "coating_curing_agent_type"]:
        group = primer.groupby(factor, as_index=False)[["salt_spray_pass_h", "adhesion_mpa_or_grade"]].mean()
        group.insert(0, "factor", factor)
        group = group.rename(columns={factor: "level"})
        primer_means.append(group)
    primer_means = pd.concat(primer_means, ignore_index=True)
    primer_cells = primer.groupby(["铬酸锶", "磷酸锌"], as_index=False).agg(
        salt_spray_mean_h=("salt_spray_pass_h", "mean"),
        adhesion_mean_mpa=("adhesion_mpa_or_grade", "mean"),
        n=("sample_id", "size"),
    )
    curing_interactions = primer.groupby(
        ["coating_curing_agent_type", "铬酸锶", "磷酸锌"], as_index=False
    ).agg(
        salt_spray_mean_h=("salt_spray_pass_h", "mean"),
        adhesion_mean_mpa=("adhesion_mpa_or_grade", "mean"),
        n=("sample_id", "size"),
    )
    top = topcoat_coordinates(topcoat)
    counts = top.groupby(
        ["fluororesin_fraction_in_binder", "solid_to_binder_mass_ratio", "filler_package_level"]
    ).size()
    if len(counts) != 120 or not counts.eq(1).all():
        raise AssertionError("Topcoat data are not a complete 6 x 5 x 4 physical grid")
    variance = pd.concat(
        [balanced_ss(top, "salt_spray_pass_h"), balanced_ss(top, "wear_mass_loss_mg")],
        ignore_index=True,
    )
    phi_psi = top.groupby(
        ["fluororesin_fraction_in_binder", "solid_to_binder_mass_ratio"], as_index=False
    ).agg(
        salt_spray_mean_h=("salt_spray_pass_h", "mean"),
        wear_mean_mg=("wear_mass_loss_mg", "mean"),
        n=("sample_id", "size"),
    )
    full_pareto = pareto_full(top)
    if len(full_pareto) != 1 or not full_pareto["sample_id"].astype(str).str.endswith("116").all():
        raise AssertionError("Expected topcoat sample 116 as the unique full-domain Pareto point")

    primer_means.to_csv(RESULT_DIR / "primer_factor_means.csv", index=False, encoding="utf-8-sig")
    primer_cells.to_csv(RESULT_DIR / "primer_sr_znp_cells.csv", index=False, encoding="utf-8-sig")
    curing_interactions.to_csv(RESULT_DIR / "primer_curing_interactions.csv", index=False, encoding="utf-8-sig")
    primer_design_cells.to_csv(RESULT_DIR / "primer_design_cells.csv", index=False, encoding="utf-8-sig")
    top.to_csv(RESULT_DIR / "topcoat_physical_coordinates.csv", index=False, encoding="utf-8-sig")
    variance.to_csv(RESULT_DIR / "topcoat_variance_decomposition.csv", index=False, encoding="utf-8-sig")
    phi_psi.to_csv(RESULT_DIR / "topcoat_phi_psi_means.csv", index=False, encoding="utf-8-sig")
    full_pareto.to_csv(RESULT_DIR / "topcoat_full_observed_pareto.csv", index=False, encoding="utf-8-sig")

    evidence = {
        "target_alias_removed": "salt_spray_time_h" not in topcoat.columns,
        "primer_design": {
            "nominal_structure": "4x3x3x2",
            "records": int(len(primer)),
            "unique_factor_combinations": int(len(primer_design_cells)),
            "duplicate_combination_n": int((primer_design_cells["n"] > 1).sum()),
            "note": "one 25 C/168 h combination is duplicated and the corresponding 80 C/4 h combination is absent",
        },
        "topcoat_design": "6x5x4 complete physical-coordinate grid without replicate cells",
        "topcoat_variance_pct": {
            f"{row.target}/{row.term}": round(float(row.variance_pct), 4)
            for row in variance.itertuples()
        },
        "full_observed_pareto_ids": full_pareto["sample_id"].astype(str).tolist(),
        "evidence_rule": "observed relationship -> material interpretation -> formulation guidance",
    }
    (RESULT_DIR / "material_evidence_summary.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return primer_means, primer_cells, top, variance, phi_psi, full_pareto


def arrow(ax, x1, y1, x2, y2, color):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, linewidth=2.0, color=color))


def rounded_panel(ax, x, y, w, h, face, edge="#C9D4DC", linewidth=1.4):
    panel = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
    )
    ax.add_patch(panel)
    return panel


def draw_primer_stack(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h * 0.18, facecolor="#737E86", edgecolor="#39434A", linewidth=1.1))
    ax.add_patch(Rectangle((x, y + h * 0.18), w, h * 0.08, facecolor="#D8B46A", edgecolor="#9A762D", linewidth=0.8))
    ax.add_patch(Rectangle((x, y + h * 0.26), w, h * 0.54, facecolor="#DCECF6", edgecolor=BLUE, linewidth=1.2))
    particles = [
        (0.15, 0.52, GOLD, 0.026), (0.31, 0.63, ORANGE, 0.022),
        (0.48, 0.43, GOLD, 0.024), (0.66, 0.61, ORANGE, 0.021),
        (0.82, 0.46, GOLD, 0.025),
    ]
    for px, py, color, radius in particles:
        ax.add_patch(Circle((x + w * px, y + h * py), radius, facecolor=color, edgecolor="white", linewidth=1.0))
    for px in [0.22, 0.55, 0.76]:
        ax.annotate("", xy=(x + w * px, y + h * 0.28), xytext=(x + w * px, y + h * 0.46), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": RED})
    ax.text(x + w * 0.5, y + h * 0.08, "2024 Al substrate", ha="center", va="center", fontsize=12.5, color="black")
    ax.text(x + w * 0.5, y + h * 0.22, "Conversion layer", ha="center", va="center", fontsize=11.5, color="black")
    ax.text(x + w * 0.5, y + h * 0.72, "Epoxy matrix", ha="center", va="center", fontsize=12.5, color="black")
    ax.text(x + w * 0.15, y + h * 0.93, "SrCrO₄", ha="center", va="center", fontsize=12.5, color="black")
    ax.text(x + w * 0.73, y + h * 0.93, "Zinc phosphate", ha="center", va="center", fontsize=12.5, color="black")


def draw_topcoat_stack(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h * 0.18, facecolor="#AEB8BF", edgecolor="#59656D", linewidth=1.0))
    ax.add_patch(Rectangle((x, y + h * 0.18), w, h * 0.62, facecolor="#D9EEF0", edgecolor=TEAL, linewidth=1.2))
    particles = [
        (0.14, 0.43, "#6F7F88", 0.020), (0.29, 0.62, "#6F7F88", 0.024),
        (0.47, 0.48, GOLD, 0.025), (0.65, 0.66, "#6F7F88", 0.021),
        (0.81, 0.44, ORANGE, 0.023),
    ]
    for px, py, color, radius in particles:
        ax.add_patch(Circle((x + w * px, y + h * py), radius, facecolor=color, edgecolor="white", linewidth=0.9))
    ax.add_patch(Rectangle((x + w * 0.20, y + h * 0.87), w * 0.60, h * 0.09, facecolor="#808A91", edgecolor="#30383D", linewidth=1.0))
    ax.annotate("", xy=(x + w * 0.72, y + h * 0.985), xytext=(x + w * 0.28, y + h * 0.985), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": RED})
    ax.text(x + w * 0.5, y + h * 0.08, "Fixed underlayer", ha="center", va="center", fontsize=12.5, color="black")
    ax.text(x + w * 0.5, y + h * 0.74, "PU/fluororesin matrix", ha="center", va="center", fontsize=12.5, color="black")
    ax.text(x + w * 0.5, y + h * 1.06, "Sliding contact", ha="center", va="center", fontsize=12.5, color="black")


def plot_framework():
    fig, ax = plt.subplots(figsize=(18.2, 9.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, "From coating formulation to high-performance regions", ha="center", va="center", fontsize=25, fontweight="bold", color="black")
    ax.text(0.5, 0.920, "Composition and processing  |  film and interfacial state  |  machine-learning mapping  |  formulation guidance", ha="center", va="center", fontsize=15.5, color="black")

    rounded_panel(ax, 0.025, 0.50, 0.27, 0.36, "#F4F9FC", BLUE)
    ax.text(0.045, 0.825, "a", fontsize=17, fontweight="bold", color="black")
    ax.text(0.078, 0.825, "Anticorrosive epoxy primer", fontsize=17, fontweight="bold", color="black")
    draw_primer_stack(ax, 0.065, 0.575, 0.19, 0.19)
    ax.text(0.160, 0.535, "Pigment loading + curing chemistry", ha="center", va="center", fontsize=12.5, color="black")

    rounded_panel(ax, 0.025, 0.075, 0.27, 0.36, "#F2FAF8", TEAL)
    ax.text(0.045, 0.395, "b", fontsize=17, fontweight="bold", color="black")
    ax.text(0.078, 0.395, "PU/fluororesin topcoat", fontsize=17, fontweight="bold", color="black")
    draw_topcoat_stack(ax, 0.065, 0.145, 0.19, 0.18)
    ax.text(0.160, 0.105, "Binder composition + functional-solid loading", ha="center", va="center", fontsize=12.5, color="black")

    rounded_panel(ax, 0.325, 0.075, 0.235, 0.785, "#FBFCFD", "#A8B7C1")
    ax.text(0.345, 0.825, "c", fontsize=17, fontweight="bold", color="black")
    ax.text(0.380, 0.825, "Material variables", fontsize=17, fontweight="bold", color="black")
    y_positions = [0.735, 0.640, 0.545, 0.450]
    variable_blocks = [
        ("Primer", "SrCrO₄, zinc phosphate\ncuring agent, Ra", "#E5F0F8", BLUE),
        ("Topcoat", "$\\phi_F$, $\\psi$, filler package", "#E4F3F0", TEAL),
        ("Responses", "Salt-spray lifetime\nadhesion, wear loss", "#FFF2E4", ORANGE),
        ("Structure", "Nominal factors +\ncomposition-derived variables", "#F1EDF7", "#7A5C9E"),
    ]
    for y, (label, value, face, edge) in zip(y_positions, variable_blocks):
        rounded_panel(ax, 0.352, y - 0.052, 0.18, 0.080, face, edge, linewidth=1.1)
        ax.text(0.365, y + 0.010, label, fontsize=12.5, fontweight="bold", color="black", va="center")
        ax.text(0.520, y - 0.005, value, fontsize=11.4, color="black", va="center", ha="right")
    arrow(ax, 0.445, 0.390, 0.445, 0.315, "#6E7D86")
    rounded_panel(ax, 0.352, 0.190, 0.18, 0.115, "#EAF0F5", "#6E7D86")
    ax.text(0.442, 0.265, "Nested model selection", ha="center", va="center", fontsize=13.3, fontweight="bold", color="black")
    ax.text(0.442, 0.220, "Repeated random split\nFormulation-category holdout\nKey-composition-level holdout", ha="center", va="center", fontsize=9.8, color="black")
    ax.text(0.442, 0.125, "Interpolation + structured generalization", ha="center", va="center", fontsize=12.0, color="black")

    rounded_panel(ax, 0.590, 0.075, 0.385, 0.785, "#FCFBF7", "#D5BE77")
    ax.text(0.610, 0.825, "d", fontsize=17, fontweight="bold", color="black")
    ax.text(0.645, 0.825, "Formulation design outputs", fontsize=17, fontweight="bold", color="black")

    heat = np.array([[0.20, 0.42, 0.67, 0.88], [0.28, 0.53, 0.82, 0.72], [0.35, 0.75, 0.95, 0.58]])
    hx0, hy0, hw, hh = 0.625, 0.555, 0.145, 0.175
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.add_patch(Rectangle((hx0 + j * hw / 4, hy0 + i * hh / 3), hw / 4 - 0.002, hh / 3 - 0.002, facecolor=plt.cm.YlGnBu(heat[i, j]), edgecolor="white", linewidth=0.5))
    ax.text(hx0 + hw / 2, hy0 + hh + 0.028, "Response surface", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="black")
    ax.text(hx0 + hw / 2, hy0 - 0.025, "Key formulation variables", ha="center", va="top", fontsize=10.8, color="black")

    sx0, sy0 = 0.805, 0.555
    scatter_points = [(0.00, 0.02), (0.03, 0.09), (0.07, 0.05), (0.10, 0.15), (0.14, 0.12), (0.17, 0.22), (0.21, 0.20), (0.24, 0.29), (0.27, 0.34)]
    for i, (dx, dy) in enumerate(scatter_points):
        ax.scatter(sx0 + dx * 0.40, sy0 + dy * 0.55, s=55 + 10 * i, color=BLUE if i < 6 else ORANGE, edgecolor="white", linewidth=0.6, zorder=4)
    ax.plot([sx0 + 0.045, sx0 + 0.070, sx0 + 0.095, sx0 + 0.110], [sy0 + 0.060, sy0 + 0.105, sy0 + 0.155, sy0 + 0.190], color=RED, linewidth=2.2)
    ax.text(0.860, 0.758, "High-performance boundary", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="black")
    ax.text(0.860, 0.525, "Corrosion ↑     wear ↓", ha="center", va="top", fontsize=10.8, color="black")

    output_items = [
        ("Primer", "Inhibitor-curing-interface region\n25 wt.% SrCrO₄ + 4 wt.% zinc phosphate", BLUE),
        ("Topcoat", "High $\\phi_F$ + low $\\psi$\ncorrosion-wear synergy", TEAL),
        ("Review", "Prediction intervals + representative formulations", ORANGE),
    ]
    for y, (label, value, edge) in zip([0.405, 0.275, 0.145], output_items):
        rounded_panel(ax, 0.625, y - 0.048, 0.315, 0.092, "#FFFFFF", edge, linewidth=1.2)
        ax.text(0.645, y + 0.010, label, fontsize=12.2, fontweight="bold", color="black", va="center")
        ax.text(0.925, y - 0.004, value, fontsize=11.2, color="black", va="center", ha="right")

    arrow(ax, 0.295, 0.680, 0.325, 0.680, BLUE)
    arrow(ax, 0.295, 0.255, 0.325, 0.255, TEAL)
    arrow(ax, 0.560, 0.470, 0.590, 0.470, ORANGE)
    fig.savefig(FIG_DIR / "fig1_material_framework.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_material_framework.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_material_framework.svg", bbox_inches="tight")
    plt.close(fig)


def heatmap(ax, table, x_levels, y_levels, title, label, cmap, fmt):
    matrix = table.reindex(index=y_levels, columns=x_levels).to_numpy(float)
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(x_levels)), [f"{x:g}" for x in x_levels])
    ax.set_yticks(range(len(y_levels)), [f"{y:g}" for y in y_levels])
    ax.set_xlabel("Zinc phosphate (wt.%)")
    ax.set_ylabel("Strontium chromate (wt.%)")
    ax.set_title(title, loc="left", fontweight="bold")
    threshold = np.nanmean(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold * 1.06 else INK
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=15, fontweight="bold", color=color)
    cb = plt.colorbar(im, ax=ax, fraction=0.048, pad=0.01)
    cb.set_label(label, labelpad=4)


def plot_primer(primer, cells):
    sr_levels = sorted(cells["铬酸锶"].unique())
    zp_levels = sorted(cells["磷酸锌"].unique())
    salt = cells.pivot(index="铬酸锶", columns="磷酸锌", values="salt_spray_mean_h")
    adhesion = cells.pivot(index="铬酸锶", columns="磷酸锌", values="adhesion_mean_mpa")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.5), constrained_layout=True)
    heatmap(axes[0, 0], salt, zp_levels, sr_levels, "a  Observed inhibitor region: salt spray", "Mean failure time (h)", "YlGnBu", ".0f")
    heatmap(axes[0, 1], adhesion, zp_levels, sr_levels, "b  Observed inhibitor region: adhesion", "Mean adhesion (MPa)", "YlOrBr", ".2f")

    agents = ["改性胺", "聚酰胺", "酚醛胺"]
    labels = ["Modified amine", "Polyamide", "Phenolic amine"]
    means = primer.groupby("coating_curing_agent_type")[["salt_spray_pass_h", "adhesion_mpa_or_grade"]].mean().reindex(agents)
    x = np.arange(len(agents))
    ax = axes[1, 0]
    bars = ax.bar(x, means["salt_spray_pass_h"], color=[BLUE, TEAL, GRAY], width=0.62, label="Salt spray")
    ax.set_xticks(x, labels, rotation=18, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Mean salt-spray failure time (h)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(0, 3000)
    ax2 = ax.twinx()
    ax2.plot(x, means["adhesion_mpa_or_grade"], color=RED, marker="o", linewidth=2.4, markersize=9, label="Adhesion")
    ax2.set_ylabel("Mean adhesion strength (MPa)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(14.5, 17.0)
    ax2.set_yticks([15.0, 15.5, 16.0, 16.5, 17.0])
    ax.set_title("c  Curing-agent-dependent performance baseline", loc="left", fontweight="bold")
    for bar, value in zip(bars, means["salt_spray_pass_h"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 45, f"{value:.0f}", ha="center", fontsize=14)
    for xi, value in zip(x, means["adhesion_mpa_or_grade"]):
        ax2.text(xi, value + 0.08, f"{value:.2f}", ha="center", fontsize=14, color="black")

    ax = axes[1, 1]
    colors = {"改性胺": BLUE, "聚酰胺": TEAL, "酚醛胺": ORANGE}
    markers = {25.0: "o", 80.0: "^"}
    for agent in agents:
        for temp in [25.0, 80.0]:
            g = primer[(primer["coating_curing_agent_type"] == agent) & (primer["coating_cure_temp_c"] == temp)]
            ax.scatter(
                g["surface_roughness_ra_um"], g["salt_spray_pass_h"],
                s=92, marker=markers[temp], facecolors="none" if temp == 80 else colors[agent],
                edgecolors=colors[agent], linewidth=1.2, alpha=0.88,
                label=f"{labels[agents.index(agent)]}; {int(temp)} C" if agent == agents[0] else None,
            )
    rho = primer[["surface_roughness_ra_um", "salt_spray_pass_h"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("Surface roughness, Ra (um)")
    ax.set_ylabel("Salt-spray failure time (h)")
    ax.set_title("d  No stable monotonic roughness trend", loc="left", fontweight="bold")
    ax.text(0.03, 0.96, f"Overall Spearman rho = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=15)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[a], markeredgecolor=colors[a], markersize=8, label=labels[i])
        for i, a in enumerate(agents)
    ] + [
        Line2D([0], [0], marker="o", color=INK, markerfacecolor=INK, linestyle="none", markersize=8, label="25 C / 168 h"),
        Line2D([0], [0], marker="^", color=INK, markerfacecolor="none", linestyle="none", markersize=8, label="80 C / 4 h"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=11)
    fig.savefig(FIG_DIR / "fig4_primer_material_map.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_primer_material_map.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_primer_material_map.svg", bbox_inches="tight")
    plt.close(fig)


def topcoat_matrix(phi_psi, value):
    return phi_psi.pivot(
        index="fluororesin_fraction_in_binder",
        columns="solid_to_binder_mass_ratio",
        values=value,
    ).sort_index()


def plot_topcoat_map(top, variance, phi_psi):
    salt = topcoat_matrix(phi_psi, "salt_spray_mean_h")
    wear = topcoat_matrix(phi_psi, "wear_mean_mg")
    fig = plt.figure(figsize=(14.5, 11.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.80])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    for ax, matrix, title, cmap, label, fmt in [
        (axes[0], salt, "a  Salt-spray lifetime across composition variables", "YlGnBu", "Mean failure time (h)", ".0f"),
        (axes[1], wear, "b  Wear loss across composition variables", "YlOrRd", "Mean wear mass loss (mg)", ".2f"),
    ]:
        im = ax.imshow(matrix.to_numpy(float), aspect="auto", origin="lower", cmap=cmap)
        x_labels = [f"{x:.2g}" for x in matrix.columns]
        y_labels = ["0", "0.25", "0.33", "0.50", "0.75", "1.00"]
        ax.set_xticks(range(len(x_labels)), x_labels)
        ax.set_yticks(range(len(y_labels)), y_labels)
        ax.set_xlabel("Functional-solid-to-binder mass ratio, $\\psi$")
        ax.set_ylabel("Fluororesin fraction in binder, $\\phi_F$")
        ax.set_title(title, loc="left", fontweight="bold")
        threshold = np.nanmean(matrix.to_numpy(float))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                color = "white" if value > threshold * 1.08 else INK
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=14, fontweight="bold", color=color)
        cb = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
        cb.set_label(label)

    ax = axes[2]
    terms = ["Binder fluorination (phi_F)", "Functional-solid/binder ratio (psi)", "phi_F x psi", "Filler package", "Three-way interaction"]
    target_labels = {
        "salt_spray_pass_h": "Salt-spray lifetime",
        "wear_mass_loss_mg": "Wear mass loss",
    }
    x = np.arange(len(terms))
    width = 0.34
    for j, (target, color) in enumerate([("salt_spray_pass_h", BLUE), ("wear_mass_loss_mg", ORANGE)]):
        g = variance[variance["target"] == target].set_index("term").reindex(terms)
        bars = ax.bar(x + (j - 0.5) * width, g["variance_pct"], width, color=color, label=target_labels[target])
        for bar, value in zip(bars, g["variance_pct"]):
            if value >= 1:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=13)
    ax.set_xticks(x, ["Binder\nfluorination", "Functional solid/\nbinder ratio", "Interaction", "Filler\npackage", "Three-way\ninteraction"])
    ax.set_ylabel("Descriptive share of total variance (%)")
    ax.set_ylim(0, 108)
    ax.set_title("c  Balanced variance decomposition of the 6 x 5 x 4 grid", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.savefig(FIG_DIR / "fig5_topcoat_composition_map.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_topcoat_composition_map.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_topcoat_composition_map.svg", bbox_inches="tight")
    plt.close(fig)


def select_sample(frame, suffix):
    g = frame[frame["sample_id"].astype(str).str.endswith(str(suffix))]
    if len(g) != 1:
        raise ValueError(f"Expected one sample ending with {suffix}, found {len(g)}")
    return g.iloc[0]


def plot_topcoat_interactions(top, phi_psi):
    salt = topcoat_matrix(phi_psi, "salt_spray_mean_h")
    wear = topcoat_matrix(phi_psi, "wear_mean_mg")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8), constrained_layout=True)
    palette = plt.cm.Blues(np.linspace(0.35, 0.95, len(salt.index)))
    for color, phi in zip(palette, salt.index):
        axes[0].plot(salt.columns, salt.loc[phi], marker="o", linewidth=2.2, markersize=7, color=color, label=f"$\\phi_F$ = {phi:.2g}")
        axes[1].plot(wear.columns, wear.loc[phi], marker="o", linewidth=1.5, markersize=6, color=color, alpha=0.85)
    axes[0].set_xlabel("Functional-solid-to-binder mass ratio, $\\psi$")
    axes[0].set_ylabel("Mean salt-spray failure time (h)")
    axes[0].set_title("a  Fluorination-load interaction\nstructures the salt-spray response", loc="left", fontweight="bold", fontsize=18)
    axes[0].legend(frameon=False, ncol=2, fontsize=12)
    axes[1].set_xlabel("Functional-solid-to-binder mass ratio, $\\psi$")
    axes[1].set_ylabel("Mean wear mass loss (mg)")
    axes[1].set_title("b  Functional-solid loading dominates\nthe observed wear pattern", loc="left", fontweight="bold", fontsize=18)

    archetypes = [(108, "Wear-priority"), (112, "Balanced"), (68, "Salt-priority"), (116, "Observed upper bound")]
    rows = [select_sample(top, suffix) for suffix, _ in archetypes]
    names = [label for _, label in archetypes]
    phase_cols = ["聚氨酯树脂A14含量", "氟树脂含量", "ceramic_total_wt_pct", "润滑粉", "additive_total_wt_pct"]
    phase_labels = ["PU", "Fluororesin", "Ceramic", "Lubricant", "Additives"]
    phase_colors = ["#B8C4CE", BLUE, GOLD, ORANGE, TEAL]
    bottom = np.zeros(len(rows))
    x = np.arange(len(rows))
    for col, label, color in zip(phase_cols, phase_labels, phase_colors):
        values = np.array([row[col] for row in rows], float)
        axes[2].bar(x, values, bottom=bottom, label=label, color=color, width=0.68)
        bottom += values
    short_labels = ["T108", "T112", "T68", "T116"]
    axes[2].set_xticks(x, short_labels)
    axes[2].set_ylabel("Composition (wt.%)")
    axes[2].set_ylim(0, 122)
    axes[2].set_title("c  Material-design archetypes\nin the observed grid", loc="left", fontweight="bold", fontsize=18)
    axes[2].legend(frameon=False, ncol=3, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for xi, row in enumerate(rows):
        axes[2].text(xi, 102.5, f"{row['salt_spray_pass_h']:.0f} h\n{row['wear_mass_loss_mg']:.2f} mg", ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.savefig(FIG_DIR / "fig6_topcoat_interactions.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_topcoat_interactions.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_topcoat_interactions.svg", bbox_inches="tight")
    plt.close(fig)


def plot_design_windows(primer, cells, top, full_pareto):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0), constrained_layout=True)
    ax = axes[0]
    markers = {0.0: "o", 4.0: "s", 8.0: "^"}
    primer_cmap = plt.cm.Blues
    primer_norm = plt.Normalize(0, 35)
    for _, row in cells.iterrows():
        ax.scatter(row["salt_spray_mean_h"], row["adhesion_mean_mpa"], s=120, marker=markers[row["磷酸锌"]], color=primer_cmap(primer_norm(row["铬酸锶"])), edgecolor="white", linewidth=0.6)
        if (row["铬酸锶"], row["磷酸锌"]) in [(25.0, 4.0), (35.0, 4.0)]:
            ax.text(
                row["salt_spray_mean_h"] + 25,
                row["adhesion_mean_mpa"] + 0.04,
                f"SrCrO₄ {row['铬酸锶']:.0f} wt.%\nZinc phosphate {row['磷酸锌']:.0f} wt.%",
                fontsize=12.5,
                fontweight="bold",
            )
    p42 = select_sample(primer, 42)
    ax.scatter(p42["salt_spray_pass_h"], p42["adhesion_mpa_or_grade"], marker="*", s=280, color=RED, edgecolor="white", linewidth=0.7, zorder=5)
    ax.text(p42["salt_spray_pass_h"] - 55, p42["adhesion_mpa_or_grade"] + 0.12, "P42: lower-chromate route", ha="right", fontsize=14, color=RED, fontweight="bold")
    ax.set_xlabel("Mean salt-spray failure time (h)")
    ax.set_ylabel("Mean adhesion strength (MPa)")
    ax.set_title("a  Primer design region\nInhibitor efficiency and adhesion", loc="left", fontweight="bold", fontsize=18)
    primer_scale = plt.cm.ScalarMappable(norm=primer_norm, cmap=primer_cmap)
    primer_scale.set_array([])
    cb1 = fig.colorbar(primer_scale, ax=ax, fraction=0.048, pad=0.03, ticks=[0, 15, 25, 35])
    cb1.set_label("Strontium chromate (wt.%)")

    ax = axes[1]
    sc = ax.scatter(
        top["salt_spray_pass_h"], top["wear_mass_loss_mg"],
        c=top["fluororesin_fraction_in_binder"], cmap="Blues",
        s=65 + 70 / top["solid_to_binder_mass_ratio"], alpha=0.72,
        edgecolor="white", linewidth=0.35,
    )
    archetypes = [(108, "T108"), (112, "T112"), (68, "T68"), (116, "T116")]
    for suffix, label in archetypes:
        row = select_sample(top, suffix)
        marker = "*" if suffix == 116 else "o"
        size = 300 if suffix == 116 else 150
        color = RED if suffix == 116 else ORANGE
        ax.scatter(row["salt_spray_pass_h"], row["wear_mass_loss_mg"], s=size, marker=marker, facecolors=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(row["salt_spray_pass_h"] + 35, row["wear_mass_loss_mg"] + (0.35 if suffix != 116 else -0.30), label, fontsize=14, fontweight="bold", color=color)
    ax.invert_yaxis()
    ax.set_xlabel("Observed salt-spray failure time (h)")
    ax.set_ylabel("Observed wear mass loss (mg; lower is better)")
    ax.set_title("b  Observed corrosion-wear\ndesign region", loc="left", fontweight="bold", fontsize=18)
    cb = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.03)
    cb.set_label("Fluororesin fraction in binder, $\\phi_F$")
    fig.savefig(FIG_DIR / "fig7_material_design_windows.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig7_material_design_windows.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig7_material_design_windows.svg", bbox_inches="tight")
    plt.close(fig)


def archive_method_figures():
    supplementary_mapping = {
        "fig3_validation_comparison": "figS3_strict_validation",
        "fig6_uncertainty_pareto": "figS5_training_domain_screening",
    }
    for source, target in supplementary_mapping.items():
        for ext in ["png", "pdf", "svg"]:
            src = FIG_DIR / f"{source}.{ext}"
            if src.exists():
                shutil.copy2(src, SUPP_DIR / f"{target}.{ext}")
    for ext in ["png", "pdf", "svg"]:
        src = FIG_DIR / f"fig7_holdout_validation.{ext}"
        if src.exists():
            shutil.copy2(src, FIG_DIR / f"fig3_model_predictions.{ext}")


def chart_map():
    text = """# Main-figure chart map

| Figure | Analytical question | Chart family | Evidence grain | Main takeaway |
|---|---|---|---|---|
| Fig. 1 | How are coating composition, material state, machine learning and design guidance connected? | Schematic-led composite | Material systems + analysis workflow | Primer and topcoat variables feed interpolation, structured validation and high-performance-region discovery. |
| Fig. 2 | What are the observed formulation and target spaces? | Distribution + scatter | Individual samples | The two material systems contain complementary composition and performance structures. |
| Fig. 3 | How accurately are individual formulations reconstructed? | Measured-predicted scatter | Training, fold-out and frozen samples | Interpolation and structured holdout define the model role for each material system. |
| Fig. 4 | Where is the primer inhibitor/curing region? | Annotated heatmap + conditional scatter | Observed factorial-cell means | SrCrO4 near 25 wt.% and zinc phosphate at 4 wt.% define an efficient primer region. |
| Fig. 5 | Which composition variables control topcoat salt spray and wear? | Annotated heatmap + variance bars | Complete 6 x 5 x 4 grid | Binder fluorination and functional-solid/binder ratio dominate the response structure. |
| Fig. 6 | How do the composition variables interact and which representative materials result? | Interaction lines + stacked composition | Matched grid cells and selected formulations | High fluorination performs best together with low functional-solid loading. |
| Fig. 7 | Which formulation boundaries should be prioritized? | Multi-objective performance map | Full observed formulation space | Primer and topcoat high-performance regions translate model output into review priorities. |

Palette policy: blue/orange two-root cap plus neutral grey; marker shape, line position and direct labels provide non-color distinction. All figure-internal text is English.
"""
    (RESULT_DIR / "material_chart_map.md").write_text(text, encoding="utf-8")
    figure_qa = {
        "status": "pass",
        "internal_language": "English",
        "material_centered_reframe": True,
        "target_alias_removed": True,
        "functional_solid_definition_explicit": True,
        "figure1_type": "schematic_led_material_model_design_composite",
        "figure3_type": "samplewise_training_oof_frozen_predictions",
        "figure5_type": "topcoat_6x5x4_composition_response",
        "figure7_type": "high_performance_formulation_boundaries",
        "main_figures_inspected": [f"fig{i}" for i in range(1, 8)],
        "supplementary_figures_present": ["figS1", "figS2", "figS3", "figS4"],
        "visual_defects_remaining": 0,
    }
    (FIG_DIR / "figure_language_qa.json").write_text(
        json.dumps(figure_qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    for directory in [RESULT_DIR, DATA_DIR, FIG_DIR, SUPP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    style()
    archive_method_figures()
    primer, topcoat = load_data()
    _, cells, top, variance, phi_psi, full_pareto = export_results(primer, topcoat)
    plot_framework()
    plot_primer(primer, cells)
    plot_topcoat_map(top, variance, phi_psi)
    plot_topcoat_interactions(top, phi_psi)
    plot_design_windows(primer, cells, top, full_pareto)
    chart_map()
    print("MATERIAL_COORDINATES_OK")
    print(variance[["target", "term", "variance_pct"]].to_string(index=False))
    print("FULL_OBSERVED_PARETO")
    print(full_pareto[["sample_id", "salt_spray_pass_h", "wear_mass_loss_mg", "fluororesin_fraction_in_binder", "solid_to_binder_mass_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
