#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap uncertainty, strict negative controls and stable sensitivities."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from project_paths import PROJECT_ROOT as PROJECT

DATA_DIR = PROJECT / "data"
RESULT_DIR = PROJECT / "results"
SRC = PROJECT / "src/03_strict_modeling.py"
SEED = 20260823
N_BOOT = int(os.environ.get("COATING_N_BOOT", "200"))
N_PERM = int(os.environ.get("COATING_N_PERM", "100"))

spec = importlib.util.spec_from_file_location("strict_modeling", SRC)
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def design_columns(layer: str):
    return sm.PRIMER_RAW + ["coating_curing_agent_type", "formulation_family"] if layer == "primer" else sm.TOPCOAT_COMPONENTS + ["formulation_family"]


def fixed_pipeline(layer: str, rep: str, model_name: str, params: dict):
    estimator, _ = sm.model_specs()[model_name]
    pipe = sm.pipeline(layer, rep, estimator)
    pipe.set_params(**params)
    return pipe


def finite_quantile(values, coverage=0.90):
    values = np.sort(np.asarray(values, float))
    n = len(values)
    rank = min(n - 1, int(np.ceil((n + 1) * coverage)) - 1)
    return float(values[rank])


def component_perturb_topcoat(X: pd.DataFrame, feature: str, delta: float):
    out = X.copy()
    comps = sm.TOPCOAT_COMPONENTS
    arr = out[comps].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    j = comps.index(feature)
    old = arr[:, j].copy()
    new = np.clip(old + delta, 1e-6, 99.0)
    other_sum = arr.sum(axis=1) - old
    scale = np.divide(100.0 - new, other_sum, out=np.ones_like(new), where=other_sum > 1e-12)
    for k in range(arr.shape[1]):
        if k != j:
            arr[:, k] *= scale
    arr[:, j] = new
    out.loc[:, comps] = arr
    return out


def sensitivity(pipe, layer: str, X: pd.DataFrame):
    base = np.asarray(pipe.predict(X)).ravel()
    rows = []
    if layer == "primer":
        for feature in sm.PRIMER_RAW:
            x = pd.to_numeric(X[feature], errors="coerce")
            span = float(x.max() - x.min())
            if span <= 0:
                continue
            delta = 0.05 * span
            xp = X.copy()
            xp[feature] = np.clip(x + delta, x.min(), x.max())
            diff = np.asarray(pipe.predict(xp)).ravel() - base
            rows.append((feature, float(np.mean(np.abs(diff))), float(np.mean(diff)), delta, "单变量范围5%扰动"))
        categories = sorted(X["coating_curing_agent_type"].astype(str).unique())
        means = []
        for cat in categories:
            xp = X.copy()
            xp["coating_curing_agent_type"] = cat
            means.append(float(np.mean(pipe.predict(xp))))
        if len(means) > 1:
            rows.append(("coating_curing_agent_type", float(max(means) - min(means)), float(means[-1] - means[0]), np.nan, "类别切换预测均值范围"))
    else:
        for feature in sm.TOPCOAT_COMPONENTS:
            x = pd.to_numeric(X[feature], errors="coerce")
            span = float(x.max() - x.min())
            delta = max(0.5, 0.02 * span)
            xp = component_perturb_topcoat(X, feature, delta)
            diff = np.asarray(pipe.predict(xp)).ravel() - base
            rows.append((feature, float(np.mean(np.abs(diff))), float(np.mean(diff)), delta, "增量后其余组分同比缩放至100%"))
    return rows


def strict_fixed_predictions(pipe, layer: str, X: pd.DataFrame, y: np.ndarray, noise=False, permuted_y=None):
    all_true, all_pred = [], []
    schemes = sm.outer_schemes(layer, X)
    for scheme_name in ["family_logo", "level_logo"]:
        for fold_id, (tr, te, _) in enumerate(schemes[scheme_name]):
            yy = y if permuted_y is None else permuted_y
            if not noise:
                model = clone(pipe)
                model.fit(X.iloc[tr], yy[tr])
                pred = np.asarray(model.predict(X.iloc[te])).ravel()
            else:
                rep = clone(pipe.named_steps["repr"])
                rep.fit(X.iloc[tr], yy[tr])
                a = rep.transform(X.iloc[tr])
                b = rep.transform(X.iloc[te])
                scaler = StandardScaler().fit(a)
                a, b = scaler.transform(a), scaler.transform(b)
                rng = np.random.default_rng(SEED + fold_id)
                noise_all = rng.normal(size=len(X))
                a = np.column_stack([a, noise_all[tr]])
                b = np.column_stack([b, noise_all[te]])
                estimator = clone(pipe.named_steps["model"])
                estimator.fit(a, yy[tr])
                pred = np.asarray(estimator.predict(b)).ravel()
            all_true.extend(y[te])
            all_pred.extend(pred)
    return np.asarray(all_true), np.asarray(all_pred)


def metric_row(y, pred):
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic) if len(np.unique(pred)) > 1 else np.nan,
    }


def main():
    selection = pd.read_csv(RESULT_DIR / "final_model_selection.csv", encoding="utf-8-sig")
    nested = pd.read_csv(RESULT_DIR / "nested_cv_predictions.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(SEED)
    interval_rows, effect_rows, assoc_rows, neg_rows, noise_rows = [], [], [], [], []

    for _, chosen in selection.iterrows():
        layer, target = chosen["layer"], chosen["target"]
        rep, model_name = chosen["representation"], chosen["model"]
        params = json.loads(chosen["best_params"])
        train, hold = sm.prepare_target(layer, target)
        cols = design_columns(layer)
        X_train = train[cols].copy()
        X_all = pd.concat([train[cols], hold[cols]], ignore_index=True)
        y_train = train[target].to_numpy(float)
        ids_all = pd.concat([train["sample_id"], hold["sample_id"]], ignore_index=True)
        set_all = np.array(["training"] * len(train) + ["frozen_holdout"] * len(hold))
        y_all = np.concatenate([train[target].to_numpy(float), hold[target].to_numpy(float)])
        pipe = fixed_pipeline(layer, rep, model_name, params)
        pipe.fit(X_train, y_train)
        point = np.asarray(pipe.predict(X_all)).ravel()

        boot_pred = np.empty((N_BOOT, len(X_all)), dtype=float)
        boot_effects = []
        for b in range(N_BOOT):
            idx = rng.integers(0, len(train), len(train))
            model = clone(pipe)
            model.fit(X_train.iloc[idx], y_train[idx])
            boot_pred[b] = np.asarray(model.predict(X_all)).ravel()
            for feature, mean_abs, mean_signed, delta, method in sensitivity(model, layer, X_train):
                boot_effects.append(
                    {"layer": layer, "target": target, "bootstrap": b, "feature": feature,
                     "mean_abs_effect": mean_abs, "mean_signed_effect": mean_signed,
                     "perturbation": delta, "method": method}
                )

        strict_resid = nested[
            (nested["layer"] == layer) & (nested["target"] == target)
            & (nested["representation"] == rep) & (nested["model"] == model_name)
            & (nested["scheme"] == "family_logo")
        ].copy()
        q90 = finite_quantile(np.abs(strict_resid["y_true"] - strict_resid["y_pred"]), 0.90)
        p05, p95 = np.quantile(boot_pred, [0.05, 0.95], axis=0)
        for i in range(len(X_all)):
            interval_rows.append(
                {
                    "layer": layer, "target": target, "sample_id": ids_all.iloc[i],
                    "set": set_all[i], "y_true": y_all[i], "y_pred": point[i],
                    "bootstrap_mean": float(boot_pred[:, i].mean()),
                    "bootstrap_std": float(boot_pred[:, i].std(ddof=1)),
                    "bootstrap_p05": float(p05[i]), "bootstrap_p95": float(p95[i]),
                    "conformal_q90": q90, "conformal_lower": float(point[i] - q90),
                    "conformal_upper": float(point[i] + q90),
                    "bootstrap_90_covered": bool(p05[i] <= y_all[i] <= p95[i]),
                    "conformal_90_covered": bool(point[i] - q90 <= y_all[i] <= point[i] + q90),
                }
            )

        effects = pd.DataFrame(boot_effects)
        for feature, g in effects.groupby("feature"):
            med_signed = float(g["mean_signed_effect"].median())
            sign = np.sign(med_signed)
            effect_rows.append(
                {
                    "layer": layer, "target": target, "feature": feature,
                    "mean_abs_effect": float(g["mean_abs_effect"].mean()),
                    "abs_effect_p05": float(g["mean_abs_effect"].quantile(0.05)),
                    "abs_effect_p95": float(g["mean_abs_effect"].quantile(0.95)),
                    "mean_signed_effect": float(g["mean_signed_effect"].mean()),
                    "signed_effect_p05": float(g["mean_signed_effect"].quantile(0.05)),
                    "signed_effect_p95": float(g["mean_signed_effect"].quantile(0.95)),
                    "sign_stability": float((np.sign(g["mean_signed_effect"]) == sign).mean()),
                    "method": g["method"].iloc[0],
                }
            )

        raw_features = sm.PRIMER_RAW if layer == "primer" else sm.TOPCOAT_COMPONENTS
        for feature in raw_features:
            x = pd.to_numeric(X_train[feature], errors="coerce").to_numpy(float)
            rhos = []
            for _ in range(1000):
                idx = rng.integers(0, len(x), len(x))
                rho = spearmanr(x[idx], y_train[idx]).statistic
                if np.isfinite(rho):
                    rhos.append(rho)
            full_rho = float(spearmanr(x, y_train).statistic)
            assoc_rows.append(
                {
                    "layer": layer, "target": target, "feature": feature,
                    "spearman_rho": full_rho, "rho_p05": float(np.quantile(rhos, 0.05)),
                    "rho_p95": float(np.quantile(rhos, 0.95)),
                    "sign_stability": float((np.sign(rhos) == np.sign(full_rho)).mean()),
                }
            )

        obs_true, obs_pred = strict_fixed_predictions(pipe, layer, X_train, y_train)
        obs = metric_row(obs_true, obs_pred)
        for p in range(N_PERM):
            perm_y = rng.permutation(y_train)
            py, pp = strict_fixed_predictions(pipe, layer, X_train, y_train, permuted_y=perm_y)
            row = {"layer": layer, "target": target, "permutation": p, **metric_row(py, pp)}
            neg_rows.append(row)
        ny, npred = strict_fixed_predictions(pipe, layer, X_train, y_train, noise=True)
        noise = metric_row(ny, npred)
        noise_rows.append(
            {"layer": layer, "target": target, **{f"observed_{k}": v for k, v in obs.items()},
             **{f"noise_{k}": v for k, v in noise.items()}, "r2_change": noise["r2"] - obs["r2"],
             "rmse_change": noise["rmse"] - obs["rmse"]}
        )

    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(RESULT_DIR / "uncertainty_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(effect_rows).to_csv(RESULT_DIR / "stable_feature_effects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(assoc_rows).to_csv(RESULT_DIR / "stable_associations.csv", index=False, encoding="utf-8-sig")
    neg = pd.DataFrame(neg_rows)
    neg.to_csv(RESULT_DIR / "label_permutation_controls.csv", index=False, encoding="utf-8-sig")
    noise = pd.DataFrame(noise_rows)
    noise.to_csv(RESULT_DIR / "random_noise_controls.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for key, g in intervals.groupby(["layer", "target", "set"]):
        m = metric_row(g["y_true"], g["y_pred"])
        summaries.append(
            {"layer": key[0], "target": key[1], "set": key[2], **m,
             "bootstrap_90_coverage": g["bootstrap_90_covered"].mean(),
             "conformal_90_coverage": g["conformal_90_covered"].mean(),
             "mean_bootstrap_width": (g["bootstrap_p95"] - g["bootstrap_p05"]).mean(),
             "conformal_width": 2 * g["conformal_q90"].iloc[0], "n": len(g)}
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULT_DIR / "uncertainty_summary.csv", index=False, encoding="utf-8-sig")

    neg_summary = neg.groupby(["layer", "target"]).agg(
        perm_r2_mean=("r2", "mean"), perm_r2_p95=("r2", lambda x: x.quantile(0.95)),
        perm_spearman_mean=("spearman", "mean"), perm_spearman_p95=("spearman", lambda x: x.quantile(0.95)),
    ).reset_index()
    neg_summary.to_csv(RESULT_DIR / "negative_control_summary.csv", index=False, encoding="utf-8-sig")

    print("UNCERTAINTY_STABILITY_OK")
    print(summary.to_string(index=False))
    print("NEGATIVE_CONTROLS")
    print(neg_summary.to_string(index=False))
    print("NOISE_CONTROLS")
    print(noise.to_string(index=False))


if __name__ == "__main__":
    main()
