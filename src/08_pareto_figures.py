#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative Pareto screening, closure-preserving responses and main figures."""

from __future__ import annotations

import importlib.util
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from project_paths import PROJECT_ROOT as PROJECT

DATA_DIR = PROJECT / "data"
RESULT_DIR = PROJECT / "results"
FIG_DIR = PROJECT / "figure"
SEED = 20260823
FEATURE_LABELS = {
    "coating_curing_agent_type": "Curing-agent type",
    "coating_cure_temp_c": "Curing temperature",
    "coating_cure_time_h": "Curing time",
    "surface_roughness_ra_um": "Surface roughness",
    "铬酸锶": "Strontium chromate",
    "磷酸锌": "Zinc phosphate",
    "聚氨酯树脂A14含量": "Polyurethane A14",
    "氟树脂含量": "Fluororesin",
    "碳化硅": "Silicon carbide",
    "氮化硼": "Boron nitride",
    "润滑粉": "Lubricant powder",
    "分散剂": "Dispersant",
    "防沉剂": "Anti-settling agent",
    "消泡剂": "Defoamer",
}

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


def fixed_pipe(chosen, layer, target):
    estimator, _ = sm.model_specs()[chosen["model"]]
    pipe = sm.pipeline(layer, chosen["representation"], estimator)
    pipe.set_params(**json.loads(chosen["best_params"]))
    train, hold = sm.prepare_target(layer, target)
    pipe.fit(train, train[target].to_numpy(float))
    return pipe, train, hold


def pareto_mask(values):
    values = np.asarray(values, float)
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        keep[i] = not np.any(np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1))
    return keep


def minmax(s):
    s = pd.Series(s, dtype=float)
    span = s.max() - s.min()
    return (s - s.min()) / span if span > 0 else pd.Series(0.5, index=s.index)


def uncertainty_wide(layer, targets, set_name="training"):
    unc = pd.read_csv(RESULT_DIR / "uncertainty_predictions.csv", encoding="utf-8-sig")
    unc = unc[(unc["layer"] == layer) & (unc["set"] == set_name)]
    out = None
    for target in targets:
        g = unc[unc["target"] == target][[
            "sample_id", "y_true", "y_pred", "bootstrap_std", "bootstrap_p05", "bootstrap_p95",
            "conformal_lower", "conformal_upper", "bootstrap_90_covered", "conformal_90_covered",
        ]].copy()
        g = g.rename(columns={c: f"{target}__{c}" for c in g.columns if c != "sample_id"})
        out = g if out is None else out.merge(g, on="sample_id", how="inner")
    return out


def knn_distance(layer, df):
    if layer == "primer":
        rep = sm.PrimerRepresentation("raw").fit(df)
    else:
        rep = sm.TopcoatRepresentation("ilr", zero_factor=0.5).fit(df)
    z = StandardScaler().fit_transform(rep.transform(df))
    n_neighbors = min(6, len(df))
    dist, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(z).kneighbors(z)
    return dist[:, 1:].mean(axis=1)


def perturbed_predictions(pipe, layer, row, n=100):
    sid = str(row.get("sample_id", ""))
    stable_offset = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16) % 100000
    rng = np.random.default_rng(SEED + stable_offset)
    rows = pd.concat([row.to_frame().T] * n, ignore_index=True)
    if layer == "primer":
        bounds = {
            "铬酸锶": (0, 35, 0.5), "磷酸锌": (0, 8, 0.25),
            "surface_roughness_ra_um": (1.5, 3.5, 0.05),
        }
        for feature, (lo, hi, sd) in bounds.items():
            base = float(pd.to_numeric(row[feature], errors="coerce"))
            rows[feature] = np.clip(base + rng.normal(0, sd, n), lo, hi)
    else:
        comps = sm.TOPCOAT_COMPONENTS
        base = np.array([float(pd.to_numeric(row[c], errors="coerce")) for c in comps])
        arr = np.clip(base[None, :] + rng.normal(0, 0.30, size=(n, len(comps))), 1e-4, None)
        arr = arr / arr.sum(axis=1, keepdims=True) * 100.0
        rows.loc[:, comps] = arr
    pred = np.asarray(pipe.predict(rows)).ravel()
    return float(np.std(pred, ddof=1)), float(np.quantile(pred, 0.05)), float(np.quantile(pred, 0.95))


def build_layer_table(layer, objectives, minimize_second=False):
    train, _ = sm.prepare_target(layer, objectives[0])
    wide = uncertainty_wide(layer, objectives, "training")
    table = train.merge(wide, on="sample_id", how="inner", suffixes=("", "_unc"))
    table["domain_distance"] = knn_distance(layer, table)

    selection = pd.read_csv(RESULT_DIR / "final_model_selection.csv", encoding="utf-8-sig")
    sensitivity_cols = []
    for target in objectives:
        chosen = selection[(selection["layer"] == layer) & (selection["target"] == target)].iloc[0]
        pipe, _, _ = fixed_pipe(chosen, layer, target)
        vals = [perturbed_predictions(pipe, layer, row) for _, row in table.iterrows()]
        table[f"{target}__perturb_std"] = [x[0] for x in vals]
        table[f"{target}__perturb_p05"] = [x[1] for x in vals]
        table[f"{target}__perturb_p95"] = [x[2] for x in vals]
        sensitivity_cols.append(f"{target}__perturb_std")

    t1, t2 = objectives
    table[f"{t1}__robust"] = np.minimum(table[f"{t1}__bootstrap_p05"], table[f"{t1}__conformal_lower"])
    if minimize_second:
        table[f"{t2}__robust"] = np.maximum(table[f"{t2}__bootstrap_p95"], table[f"{t2}__conformal_upper"])
        values = np.column_stack([table[f"{t1}__robust"], -table[f"{t2}__robust"]])
        score2 = 1 - minmax(table[f"{t2}__robust"])
    else:
        table[f"{t2}__robust"] = np.minimum(table[f"{t2}__bootstrap_p05"], table[f"{t2}__conformal_lower"])
        values = np.column_stack([table[f"{t1}__robust"], table[f"{t2}__robust"]])
        score2 = minmax(table[f"{t2}__robust"])
    table["is_robust_pareto"] = pareto_mask(values)
    uncertainty = minmax(table[f"{t1}__bootstrap_std"]) + minmax(table[f"{t2}__bootstrap_std"])
    perturb = minmax(table[sensitivity_cols[0]]) + minmax(table[sensitivity_cols[1]])
    table["robust_score"] = (
        0.5 * minmax(table[f"{t1}__robust"]) + 0.5 * score2
        - 0.08 * uncertainty - 0.07 * minmax(table["domain_distance"]) - 0.05 * perturb
    )
    table = table.sort_values(["is_robust_pareto", "robust_score"], ascending=[False, False]).reset_index(drop=True)
    return table


def select_candidates(table, n=10):
    pareto = table[table["is_robust_pareto"]].copy()
    selected = pareto.head(n).copy()
    if len(selected) < n:
        fill = table[~table["sample_id"].isin(selected["sample_id"])].head(n - len(selected))
        selected = pd.concat([selected, fill], ignore_index=True)
    selected["candidate_rank"] = np.arange(1, len(selected) + 1)
    return pareto, selected


def screening_view(table, layer, objectives):
    design = sm.PRIMER_RAW + ["coating_curing_agent_type"] if layer == "primer" else sm.TOPCOAT_COMPONENTS
    cols = ["sample_id", "formulation_family", "is_design_boundary"] + design
    for target in objectives:
        cols.extend(
            [
                target, f"{target}__y_pred", f"{target}__bootstrap_std",
                f"{target}__bootstrap_p05", f"{target}__bootstrap_p95",
                f"{target}__conformal_lower", f"{target}__conformal_upper",
                f"{target}__robust", f"{target}__perturb_std",
            ]
        )
    cols.extend(["domain_distance", "is_robust_pareto", "robust_score"])
    if "candidate_rank" in table.columns:
        cols.insert(0, "candidate_rank")
    return table[[c for c in cols if c in table.columns]].copy()


def holdout_table(layer, objectives, minimize_second=False):
    wide = uncertainty_wide(layer, objectives, "frozen_holdout")
    hold_file = DATA_DIR / ("frozen_holdout_primer.csv" if layer == "primer" else "frozen_holdout_topcoat.csv")
    hold = pd.read_csv(hold_file, encoding="utf-8-sig")
    out = hold.merge(wide, on="sample_id", how="inner")
    t1, t2 = objectives
    true_score = minmax(out[f"{t1}__y_true"]) + (1 - minmax(out[f"{t2}__y_true"]) if minimize_second else minmax(out[f"{t2}__y_true"]))
    pred_score = minmax(out[f"{t1}__y_pred"]) + (1 - minmax(out[f"{t2}__y_pred"]) if minimize_second else minmax(out[f"{t2}__y_pred"]))
    out["true_combined_score"] = true_score
    out["predicted_combined_score"] = pred_score
    labels = (out["holdout_role"] == "高性能/Pareto").astype(int)
    out["high_performance_label"] = labels
    auc = float(roc_auc_score(labels, pred_score)) if labels.nunique() == 2 else np.nan
    out["high_performance_auc"] = auc
    return out


def normalized_effects(layer, targets):
    effects = pd.read_csv(RESULT_DIR / "stable_feature_effects.csv", encoding="utf-8-sig")
    effects = effects[(effects["layer"] == layer) & (effects["target"].isin(targets))].copy()
    effects["normalized_effect"] = effects.groupby("target")["mean_abs_effect"].transform(lambda x: x / x.max() if x.max() > 0 else x)
    return effects


def plot_primer(primer, candidates, selection):
    salt_chosen = selection[(selection["layer"] == "primer") & (selection["target"] == "salt_spray_pass_h")].iloc[0]
    salt_pipe, train, _ = fixed_pipe(salt_chosen, "primer", "salt_spray_pass_h")
    effects = normalized_effects("primer", ["salt_spray_pass_h", "adhesion_mpa_or_grade"])
    top_features = effects.groupby("feature")["normalized_effect"].max().nlargest(8).index
    e = effects[effects["feature"].isin(top_features)]

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.8), constrained_layout=True)
    ax = axes[0]
    for j, (target, color, legend_label) in enumerate([("salt_spray_pass_h", "#4C78A8", "Salt spray"), ("adhesion_mpa_or_grade", "#E45756", "Adhesion")]):
        g = e[e["target"] == target].set_index("feature").reindex(top_features)
        ax.barh(np.arange(len(top_features)) + (j - 0.5) * 0.34, g["normalized_effect"], height=0.32, color=color, label=legend_label)
    ax.set_yticks(np.arange(len(top_features)), [FEATURE_LABELS.get(x, x) for x in top_features])
    for label in ax.get_yticklabels():
        label.set_rotation(30)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.invert_yaxis()
    ax.set_xlabel("Normalized stable perturbation effect")
    ax.set_title("a  Stable primer factors", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    total_grid = np.linspace(0.5, 42, 45)
    balance_grid = np.linspace(-1, 1, 45)
    base = train.iloc[[0]].copy()
    base["coating_curing_agent_type"] = train["coating_curing_agent_type"].mode().iloc[0]
    base["coating_cure_temp_c"] = train["coating_cure_temp_c"].median()
    base["coating_cure_time_h"] = train["coating_cure_time_h"].median()
    base["surface_roughness_ra_um"] = train["surface_roughness_ra_um"].median()
    z = np.full((len(balance_grid), len(total_grid)), np.nan)
    for iy, bal in enumerate(balance_grid):
        rows = pd.concat([base] * len(total_grid), ignore_index=True)
        sr = total_grid * (1 + bal) / 2
        zp = total_grid * (1 - bal) / 2
        valid = (sr <= 35) & (zp <= 8)
        rows["铬酸锶"], rows["磷酸锌"] = sr, zp
        pred = np.asarray(salt_pipe.predict(rows)).ravel()
        z[iy, valid] = pred[valid]
    ax = axes[1]
    im = ax.imshow(z, origin="lower", aspect="auto", extent=[total_grid.min(), total_grid.max(), -1, 1], cmap="viridis")
    fig.colorbar(im, ax=ax, label="Predicted salt-spray lifetime (h)")
    ax.set_xlabel("Total inhibitor content (wt.%)")
    ax.set_ylabel("Inhibitor balance (Sr-Zn)/(Sr+Zn)")
    ax.set_title("b  Physics-informed response surface", loc="left", fontweight="bold")

    ax = axes[2]
    ax.scatter(primer["salt_spray_pass_h__robust"], primer["adhesion_mpa_or_grade__robust"], c="#B8C4CE", s=50, label="Training candidates")
    p = primer[primer["is_robust_pareto"]]
    ax.scatter(p["salt_spray_pass_h__robust"], p["adhesion_mpa_or_grade__robust"], c="#F58518", s=75, label="Conservative Pareto")
    ax.scatter(candidates["salt_spray_pass_h__robust"], candidates["adhesion_mpa_or_grade__robust"], facecolors="none", edgecolors="#C62828", s=130, linewidth=2.0, label="Robust shortlist")
    ax.set_xlabel("Conservative salt-spray lower bound (h)")
    ax.set_ylabel("Conservative adhesion lower bound (MPa)")
    ax.set_title("c  Conservative primer Pareto front", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "fig4_primer_explanation.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_primer_explanation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_topcoat(topcoat, candidates, selection):
    effects = normalized_effects("topcoat", ["salt_spray_pass_h", "wear_mass_loss_mg"])
    top_features = effects.groupby("feature")["normalized_effect"].max().nlargest(8).index
    e = effects[effects["feature"].isin(top_features)]
    salt_chosen = selection[(selection["layer"] == "topcoat") & (selection["target"] == "salt_spray_pass_h")].iloc[0]
    wear_chosen = selection[(selection["layer"] == "topcoat") & (selection["target"] == "wear_mass_loss_mg")].iloc[0]
    salt_pipe, train, _ = fixed_pipe(salt_chosen, "topcoat", "salt_spray_pass_h")
    wear_pipe, _, _ = fixed_pipe(wear_chosen, "topcoat", "wear_mass_loss_mg")

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.8), constrained_layout=True)
    ax = axes[0]
    for j, (target, color, legend_label) in enumerate([("salt_spray_pass_h", "#4C78A8", "Salt spray"), ("wear_mass_loss_mg", "#E45756", "Wear loss")]):
        g = e[e["target"] == target].set_index("feature").reindex(top_features)
        ax.barh(np.arange(len(top_features)) + (j - 0.5) * 0.34, g["normalized_effect"], height=0.32, color=color, label=legend_label)
    ax.set_yticks(np.arange(len(top_features)), [FEATURE_LABELS.get(x, x) for x in top_features])
    for label in ax.get_yticklabels():
        label.set_rotation(30)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.invert_yaxis()
    ax.set_xlabel("Normalized closure-preserving effect")
    ax.set_title("a  Stable topcoat factors", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    pu = pd.to_numeric(train["聚氨酯树脂A14含量"], errors="coerce")
    fluor = pd.to_numeric(train["氟树脂含量"], errors="coerce")
    fractions = fluor / (pu + fluor).replace(0, np.nan)
    observed = pd.DataFrame(
        {
            "fraction": fractions,
            "salt": pd.to_numeric(train["salt_spray_pass_h"], errors="coerce"),
            "wear": pd.to_numeric(train["wear_mass_loss_mg"], errors="coerce"),
        }
    ).dropna()
    observed["bin"] = pd.qcut(observed["fraction"].rank(method="first"), q=6, labels=False)
    trend = observed.groupby("bin", as_index=False).agg(fraction=("fraction", "median"), salt=("salt", "median"), wear=("wear", "median"))
    ax = axes[1]
    ax.scatter(observed["fraction"], observed["salt"], color="#4C78A8", alpha=0.25, s=36)
    ax.plot(trend["fraction"], trend["salt"], color="#4C78A8", marker="o", linewidth=2.2, markersize=8, label="Binned salt-spray median")
    ax.set_xlabel("Fluororesin fraction within binder phase")
    ax.set_ylabel("Measured salt-spray lifetime (h)", color="#4C78A8")
    ax.tick_params(axis="y", labelcolor="#4C78A8")
    ax2 = ax.twinx()
    ax2.scatter(observed["fraction"], observed["wear"], color="#E45756", alpha=0.18, s=36)
    ax2.plot(trend["fraction"], trend["wear"], color="#E45756", marker="s", linewidth=2.2, markersize=8, label="Binned wear median")
    ax2.set_ylabel("Measured wear mass loss (mg)", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    ax2.spines["top"].set_visible(False)
    ax.set_title("b  Binder-balance trend within observed domain", loc="left", fontweight="bold")

    ax = axes[2]
    ax.scatter(topcoat["salt_spray_pass_h__robust"], topcoat["wear_mass_loss_mg__robust"], c="#B8C4CE", s=100, label="Training candidates")
    p = topcoat[topcoat["is_robust_pareto"]]
    ax.scatter(p["salt_spray_pass_h__robust"], p["wear_mass_loss_mg__robust"], c="#F58518", s=150, label="Conservative Pareto")
    ax.scatter(candidates["salt_spray_pass_h__robust"], candidates["wear_mass_loss_mg__robust"], facecolors="none", edgecolors="#C62828", s=260, linewidth=2.0, label="Robust shortlist")
    ax.invert_yaxis()
    ax.set_xlabel("Conservative salt-spray lower bound (h)")
    ax.set_ylabel("Conservative wear upper bound (mg; lower is better)")
    ax.set_title("c  Conservative topcoat Pareto front", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "fig5_topcoat_tradeoff.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_topcoat_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pareto(primer, topcoat, p_candidates, m_candidates):
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.2), constrained_layout=True)
    ax = axes[0]
    ax.scatter(primer["salt_spray_pass_h__robust"], primer["adhesion_mpa_or_grade__robust"], c=primer["domain_distance"], cmap="Blues", s=68)
    ax.scatter(p_candidates["salt_spray_pass_h__robust"], p_candidates["adhesion_mpa_or_grade__robust"], facecolors="none", edgecolors="#C62828", s=180, linewidth=1.3)
    ax.set_xlabel("Conservative salt-spray lower bound (h)")
    ax.set_ylabel("Conservative adhesion lower bound (MPa)")
    ax.set_title("a  Primer: corrosion-adhesion screening", loc="left", fontweight="bold")
    ax = axes[1]
    sc = ax.scatter(topcoat["salt_spray_pass_h__robust"], topcoat["wear_mass_loss_mg__robust"], c=topcoat["domain_distance"], cmap="Blues", s=68)
    ax.scatter(m_candidates["salt_spray_pass_h__robust"], m_candidates["wear_mass_loss_mg__robust"], facecolors="none", edgecolors="#C62828", s=180, linewidth=1.3)
    ax.invert_yaxis()
    ax.set_xlabel("Conservative salt-spray lower bound (h)")
    ax.set_ylabel("Conservative wear upper bound (mg)")
    ax.set_title("b  Topcoat: corrosion-wear screening", loc="left", fontweight="bold")
    fig.colorbar(sc, ax=axes, label="Local design-space distance")
    fig.savefig(FIG_DIR / "fig6_uncertainty_pareto.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_uncertainty_pareto.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_uncertainty_pareto.svg", bbox_inches="tight")
    plt.close(fig)


def plot_holdout():
    configs = [
        ("primer", "salt_spray_pass_h", "a  Primer salt-spray lifetime (h)"),
        ("primer", "adhesion_mpa_or_grade", "b  Primer adhesion strength (MPa)"),
        ("topcoat", "salt_spray_pass_h", "c  Topcoat salt-spray lifetime (h)"),
        ("topcoat", "wear_mass_loss_mg", "d  Topcoat wear mass loss (mg)"),
    ]
    unc = pd.read_csv(RESULT_DIR / "uncertainty_predictions.csv", encoding="utf-8-sig")
    nested = pd.read_csv(RESULT_DIR / "nested_cv_predictions.csv", encoding="utf-8-sig")
    selected = pd.read_csv(RESULT_DIR / "final_model_selection.csv", encoding="utf-8-sig")
    fig = plt.figure(figsize=(14.5, 13.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.10, 1.0, 1.0])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    axes = np.array(
        [
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
            [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])],
        ]
    )
    legend_handles = None
    for ax, (layer, target, title) in zip(axes.ravel(), configs):
        chosen = selected[(selected["layer"] == layer) & (selected["target"] == target)].iloc[0]
        fit = unc[(unc["layer"] == layer) & (unc["target"] == target) & (unc["set"] == "training")]
        frozen = unc[(unc["layer"] == layer) & (unc["target"] == target) & (unc["set"] == "frozen_holdout")]
        oof = nested[
            (nested["layer"] == layer)
            & (nested["target"] == target)
            & (nested["representation"] == chosen["representation"])
            & (nested["model"] == chosen["model"])
            & (nested["scheme"] == "family_logo")
        ]
        if fit.empty or oof.empty or frozen.empty:
            raise ValueError(f"Missing validation group for {layer}/{target}")

        fit_sc = ax.scatter(
            fit["y_true"], fit["y_pred"], s=64, marker="o", color="#B8C0C8",
            alpha=0.58, edgecolor="white", linewidth=0.35, label="Model-fit training", zorder=2,
        )
        oof_sc = ax.scatter(
            oof["y_true"], oof["y_pred"], s=92, marker="^", facecolors="none",
            alpha=0.88, edgecolor="#4C78A8", linewidth=1.05,
            label="Formulation-category OOF test", zorder=3,
        )
        frozen_sc = ax.scatter(
            frozen["y_true"], frozen["y_pred"], s=124, marker="o", color="#C62828",
            edgecolor="white", linewidth=0.65, label="Frozen verification set", zorder=5,
        )
        ax.vlines(
            frozen["y_true"], frozen["bootstrap_p05"], frozen["bootstrap_p95"],
            color="#C62828", alpha=0.72, linewidth=1.25, zorder=4,
        )
        all_values = np.concatenate(
            [
                fit[["y_true", "y_pred"]].to_numpy(float).ravel(),
                oof[["y_true", "y_pred"]].to_numpy(float).ravel(),
                frozen[["y_true", "y_pred", "bootstrap_p05", "bootstrap_p95"]].to_numpy(float).ravel(),
            ]
        )
        low, high = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        padding = max(0.05 * (high - low), 1e-9)
        low, high = low - padding, high + padding
        ax.plot([low, high], [low, high], "--", color="#444444", linewidth=1.0, zorder=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Measured value")
        ax.set_ylabel("Predicted value")
        ax.set_title(title, loc="left", fontweight="bold")
        coverage = frozen["bootstrap_90_covered"].mean()
        ax.text(
            0.04, 0.95,
            f"n(fit/OOF/frozen) = {len(fit)}/{len(oof)}/{len(frozen)}\nFrozen-set 90% coverage = {coverage:.2f}",
            transform=ax.transAxes, va="top", fontsize=14,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
            zorder=6,
        )
        if legend_handles is None:
            legend_handles = [fit_sc, oof_sc, frozen_sc]
    legend_ax.legend(
        handles=legend_handles,
        labels=["Model-fit training", "Formulation-category OOF test", "Frozen verification set"],
        loc="center", ncol=3, frameon=False,
    )
    fig.savefig(FIG_DIR / "fig7_holdout_validation.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig7_holdout_validation.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig7_holdout_validation.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(primer, topcoat, p_candidates, m_candidates, holdout, selection):
    agg = pd.read_csv(RESULT_DIR / "nested_cv_summary.csv", encoding="utf-8-sig")
    unc = pd.read_csv(RESULT_DIR / "uncertainty_summary.csv", encoding="utf-8-sig")
    neg = pd.read_csv(RESULT_DIR / "negative_control_summary.csv", encoding="utf-8-sig")
    lines = ["# 正式结果汇总", "", "## 1. 严格外推验证", "", "| 层级 | 目标 | 表示 | 模型 | 随机R² | 配方族R² | 水平留出R² |", "|---|---|---|---|---:|---:|---:|"]
    for _, s in selection.iterrows():
        q = agg[(agg["layer"] == s["layer"]) & (agg["target"] == s["target"]) & (agg["representation"] == s["representation"]) & (agg["model"] == s["model"])]
        vals = {r["scheme"]: r["pooled_r2"] for _, r in q.iterrows()}
        lines.append(f"| {s['layer']} | {s['target']} | {s['representation']} | {s['model']} | {vals.get('random_repeated', np.nan):.2f} | {vals.get('family_logo', np.nan):.2f} | {vals.get('level_logo', np.nan):.2f} |")
    lines.extend(["", "底漆在严格外推下显著弱于随机邻域内插，面漆仍近乎线性可预测。面漆结果应解释为数据结构高度规则，而不是复杂模型优势。", "", "## 2. 负对照与不确定性", ""])
    for _, r in neg.iterrows():
        lines.append(f"- {r['layer']} / {r['target']}：标签置换R² 95%分位数 {r['perm_r2_p95']:.2f}，置换Spearman 95%分位数 {r['perm_spearman_p95']:.2f}。")
    for _, r in unc[unc["set"] == "frozen_holdout"].iterrows():
        lines.append(f"- 冻结集 {r['layer']} / {r['target']}：R²={r['r2']:.2f}，Bootstrap 90%覆盖率={r['bootstrap_90_coverage']:.2f}，conformal 90%覆盖率={r['conformal_90_coverage']:.2f}。")
    lines.extend(["", "## 3. 稳健候选", "", f"- 底漆保守Pareto样本 {primer['is_robust_pareto'].sum()} 条，输出候选 {len(p_candidates)} 条。", f"- 面漆保守Pareto样本 {topcoat['is_robust_pareto'].sum()} 条，输出候选 {len(m_candidates)} 条。", f"- 冻结高性能识别AUC：底漆 {holdout[holdout['layer']=='primer']['high_performance_auc'].iloc[0]:.2f}，面漆 {holdout[holdout['layer']=='topcoat']['high_performance_auc'].iloc[0]:.2f}。", "", "候选来自训练域内的既有样本，用于确定配方复核顺序。", "", "## 4. 论文表述边界", "", "1. 底漆结论聚焦抑制剂—固化—界面作用和区间不确定性。", "2. 面漆结论聚焦闭合组成下的规则化关系，并用闭合保持扰动解释树脂/填料平衡。", "3. 结论范围与观测数据和验证方案保持一致。"])
    (RESULT_DIR / "model_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    style()
    selection = pd.read_csv(RESULT_DIR / "final_model_selection.csv", encoding="utf-8-sig")
    primer = build_layer_table("primer", ["salt_spray_pass_h", "adhesion_mpa_or_grade"], minimize_second=False)
    topcoat = build_layer_table("topcoat", ["salt_spray_pass_h", "wear_mass_loss_mg"], minimize_second=True)
    screening_view(primer, "primer", ["salt_spray_pass_h", "adhesion_mpa_or_grade"]).to_csv(RESULT_DIR / "screening_all_primer.csv", index=False, encoding="utf-8-sig")
    screening_view(topcoat, "topcoat", ["salt_spray_pass_h", "wear_mass_loss_mg"]).to_csv(RESULT_DIR / "screening_all_topcoat.csv", index=False, encoding="utf-8-sig")
    p_pareto, p_candidates = select_candidates(primer, 10)
    m_pareto, m_candidates = select_candidates(topcoat, 10)
    screening_view(p_pareto, "primer", ["salt_spray_pass_h", "adhesion_mpa_or_grade"]).to_csv(RESULT_DIR / "pareto_primer.csv", index=False, encoding="utf-8-sig")
    screening_view(m_pareto, "topcoat", ["salt_spray_pass_h", "wear_mass_loss_mg"]).to_csv(RESULT_DIR / "pareto_topcoat.csv", index=False, encoding="utf-8-sig")
    screening_view(p_candidates, "primer", ["salt_spray_pass_h", "adhesion_mpa_or_grade"]).to_csv(RESULT_DIR / "robust_candidates_primer.csv", index=False, encoding="utf-8-sig")
    screening_view(m_candidates, "topcoat", ["salt_spray_pass_h", "wear_mass_loss_mg"]).to_csv(RESULT_DIR / "robust_candidates_topcoat.csv", index=False, encoding="utf-8-sig")

    p_hold = holdout_table("primer", ["salt_spray_pass_h", "adhesion_mpa_or_grade"], False)
    m_hold = holdout_table("topcoat", ["salt_spray_pass_h", "wear_mass_loss_mg"], True)
    p_hold["layer"] = "primer"
    m_hold["layer"] = "topcoat"
    holdout = pd.concat([p_hold, m_hold], ignore_index=True, sort=False)
    holdout.to_csv(RESULT_DIR / "holdout_predictions.csv", index=False, encoding="utf-8-sig")

    plot_primer(primer, p_candidates, selection)
    plot_topcoat(topcoat, m_candidates, selection)
    plot_pareto(primer, topcoat, p_candidates, m_candidates)
    plot_holdout()
    write_summary(primer, topcoat, p_candidates, m_candidates, holdout, selection)
    print("PARETO_FIGURES_OK")
    print("PRIMER_CANDIDATES")
    print(p_candidates[["candidate_rank", "sample_id", "formulation_family", "salt_spray_pass_h__robust", "adhesion_mpa_or_grade__robust", "robust_score"]].to_string(index=False))
    print("TOPCOAT_CANDIDATES")
    print(m_candidates[["candidate_rank", "sample_id", "formulation_family", "salt_spray_pass_h__robust", "wear_mass_loss_mg__robust", "robust_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
