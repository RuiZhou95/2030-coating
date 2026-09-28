#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nested validation for independent primer and topcoat domains."""

from __future__ import annotations

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import PRIMER_FILE, PROJECT_ROOT as PROJECT, RAW_DATA_DIR as RAW_DIR, SHEET_NAME, TOPCOAT_FILE

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA_DIR = PROJECT / "data"
RESULT_DIR = PROJECT / "results"
FIG_DIR = PROJECT / "figure"
MODEL_DIR = RESULT_DIR / "models"
SEED = 20260823

FILES = {
    "primer": RAW_DIR / PRIMER_FILE,
    "topcoat": RAW_DIR / TOPCOAT_FILE,
}
TARGET_SPECS = [
    ("primer", "salt_spray_pass_h", "core"),
    ("primer", "adhesion_mpa_or_grade", "core"),
    ("primer", "wear_mass_loss_mg", "auxiliary"),
    ("topcoat", "salt_spray_pass_h", "core"),
    ("topcoat", "wear_mass_loss_mg", "core"),
]
PRIMER_RAW = ["铬酸锶", "磷酸锌", "coating_cure_temp_c", "coating_cure_time_h", "surface_roughness_ra_um"]
TOPCOAT_COMPONENTS = ["聚氨酯树脂A14含量", "氟树脂含量", "碳化硅", "氮化硼", "润滑粉", "分散剂", "防沉剂", "消泡剂"]


def clean_name(value) -> str:
    return str(value).replace("\n", "").strip()


def load_layer(layer: str) -> pd.DataFrame:
    path = FILES[layer]
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=1, skiprows=[2, 3, 4, 5, 6, 7])
    df.columns = [clean_name(c) for c in df.columns]
    if layer == "topcoat":
        raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=None, nrows=3)
        rename = {}
        for i in range(8, min(16, raw.shape[1])):
            name = clean_name(raw.iloc[2, i])
            if name and name.lower() != "nan":
                rename[f"Unnamed: {i}"] = name
        df = df.rename(columns=rename)
    df = df.dropna(subset=["sample_id"]).copy()
    df = df[~df["sample_id"].astype(str).str.contains("第10行", na=False)].copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    zh_layer = "底漆" if layer == "primer" else "面漆"
    groups = pd.read_csv(DATA_DIR / "sample_groups.csv", encoding="utf-8-sig")
    groups = groups[groups["layer"] == zh_layer][["sample_id", "formulation_family", "is_design_boundary"]]
    return df.merge(groups, on="sample_id", how="left").reset_index(drop=True)


class PrimerRepresentation(BaseEstimator, TransformerMixin):
    def __init__(self, kind="raw"):
        self.kind = kind

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.categories_ = sorted(X["coating_curing_agent_type"].fillna("Unknown").astype(str).unique())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        vals = {c: pd.to_numeric(X[c], errors="coerce").fillna(0).to_numpy(float) for c in PRIMER_RAW}
        sr, zp = vals["铬酸锶"], vals["磷酸锌"]
        if self.kind == "raw":
            cols = [vals[c] for c in PRIMER_RAW]
        elif self.kind == "physical":
            total = sr + zp
            balance = np.divide(sr - zp, total, out=np.zeros_like(total), where=total > 1e-12)
            cure_severity = vals["coating_cure_temp_c"] * np.log1p(vals["coating_cure_time_h"])
            cols = [
                total, balance, vals["coating_cure_temp_c"], vals["coating_cure_time_h"],
                vals["surface_roughness_ra_um"], cure_severity,
                total * vals["surface_roughness_ra_um"],
            ]
        else:
            raise ValueError(self.kind)
        cat = X["coating_curing_agent_type"].fillna("Unknown").astype(str)
        cols.extend([(cat == level).to_numpy(float) for level in self.categories_])
        return np.column_stack(cols)

    def get_feature_names_out(self, input_features=None):
        if self.kind == "raw":
            names = list(PRIMER_RAW)
        else:
            names = ["抑制剂总量", "抑制剂平衡", "固化温度", "固化时间", "表面粗糙度", "固化强度", "抑制剂总量×粗糙度"]
        return np.array(names + [f"固化剂={c}" for c in self.categories_], dtype=object)


class TopcoatRepresentation(BaseEstimator, TransformerMixin):
    def __init__(self, kind="raw", zero_factor=0.5):
        self.kind = kind
        self.zero_factor = zero_factor

    def fit(self, X, y=None):
        arr = self._array(X)
        positive = arr[arr > 0]
        self.delta_ = float(positive.min() * self.zero_factor) if len(positive) else 1e-6
        if self.kind == "ilr":
            self.basis_ = helmert(arr.shape[1], full=False).T
        return self

    def _array(self, X):
        X = pd.DataFrame(X).copy()
        return np.column_stack([pd.to_numeric(X[c], errors="coerce").fillna(0).to_numpy(float) for c in TOPCOAT_COMPONENTS])

    def _positive_closed(self, arr):
        z = np.where(arr <= 0, self.delta_, arr)
        return z / z.sum(axis=1, keepdims=True)

    def transform(self, X):
        arr = self._array(X)
        if self.kind == "raw":
            return arr
        comp = self._positive_closed(arr)
        if self.kind == "ilr":
            return np.log(comp) @ self.basis_
        if self.kind != "physical":
            raise ValueError(self.kind)
        pu, fluor, sic, bn, lubricant, dispersant, antisettling, defoamer = arr.T
        binder = pu + fluor
        ceramic = sic + bn
        additives = dispersant + antisettling + defoamer
        fluor_fraction = (fluor + self.delta_) / (binder + 2 * self.delta_)
        bn_fraction = (bn + self.delta_) / (ceramic + 2 * self.delta_)
        disp_fraction = (dispersant + self.delta_) / (additives + 3 * self.delta_)
        anti_fraction = (antisettling + self.delta_) / (additives + 3 * self.delta_)
        defoam_fraction = (defoamer + self.delta_) / (additives + 3 * self.delta_)
        return np.column_stack([binder, fluor_fraction, ceramic, bn_fraction, lubricant, additives, disp_fraction, anti_fraction, defoam_fraction])

    def get_feature_names_out(self, input_features=None):
        if self.kind == "raw":
            return np.array(TOPCOAT_COMPONENTS, dtype=object)
        if self.kind == "physical":
            return np.array(["树脂相总量", "树脂相中氟树脂分数", "陶瓷相总量", "陶瓷相中BN分数", "润滑相", "助剂相总量", "分散剂相对分数", "防沉剂相对分数", "消泡剂相对分数"], dtype=object)
        return np.array([f"ILR{i+1}" for i in range(len(TOPCOAT_COMPONENTS) - 1)], dtype=object)


def representation(layer: str, kind: str):
    return PrimerRepresentation(kind) if layer == "primer" else TopcoatRepresentation(kind)


def model_specs():
    specs = {
        "Dummy": (DummyRegressor(strategy="mean"), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "PLS": (PLSRegression(scale=False, max_iter=1000), {"model__n_components": [2, 3, 4]}),
        "RandomForest": (
            RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=1),
            {"model__max_depth": [3, 6, None], "model__min_samples_leaf": [1, 3]},
        ),
        "GaussianProcess": (
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(0.05),
                normalize_y=True,
                random_state=SEED,
                n_restarts_optimizer=0,
            ),
            {"model__alpha": [1e-6, 1e-3, 1e-1]},
        ),
    }
    if HAS_XGB:
        specs["XGBoost"] = (
            XGBRegressor(
                objective="reg:squarederror", random_state=SEED, n_jobs=1,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            ),
            {
                "model__n_estimators": [150, 350],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
            },
        )
    return specs


def pipeline(layer: str, rep: str, estimator):
    return Pipeline([("repr", representation(layer, rep)), ("scale", StandardScaler()), ("model", estimator)])


def safe_spearman(y, pred):
    if len(np.unique(y)) < 2 or len(np.unique(pred)) < 2:
        return np.nan
    return float(spearmanr(y, pred).statistic)


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float).ravel()
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    span = float(np.max(y) - np.min(y))
    return {
        "r2": float(r2_score(y, pred)) if len(y) > 1 and np.std(y) > 0 else np.nan,
        "rmse": rmse,
        "nrmse_range": rmse / span if span > 0 else np.nan,
        "mae": float(mean_absolute_error(y, pred)),
        "spearman": safe_spearman(y, pred),
    }


def outer_schemes(layer: str, X: pd.DataFrame):
    schemes = {}
    random_cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    schemes["random_repeated"] = [(tr, te, None) for tr, te in random_cv.split(X)]

    fam = X["formulation_family"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    schemes["family_logo"] = [(tr, te, fam[tr]) for tr, te in logo.split(X, groups=fam)]

    if layer == "primer":
        level = pd.to_numeric(X["铬酸锶"], errors="coerce").astype(str).to_numpy()
    else:
        fluor = pd.to_numeric(X["氟树脂含量"], errors="coerce")
        level = pd.qcut(fluor.rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str).to_numpy()
    schemes["level_logo"] = [(tr, te, level[tr]) for tr, te in logo.split(X, groups=level)]
    return schemes


def fit_search(pipe, grid, X, y, scheme, inner_groups, seed_offset=0):
    if scheme in {"family_logo", "level_logo"} and inner_groups is not None and len(np.unique(inner_groups)) >= 3:
        inner = GroupKFold(n_splits=min(3, len(np.unique(inner_groups))))
        search = GridSearchCV(pipe, grid or {}, scoring="neg_root_mean_squared_error", cv=inner, n_jobs=8, refit=True, error_score="raise")
        search.fit(X, y, groups=inner_groups)
    else:
        inner = KFold(n_splits=3, shuffle=True, random_state=SEED + seed_offset)
        search = GridSearchCV(pipe, grid or {}, scoring="neg_root_mean_squared_error", cv=inner, n_jobs=8, refit=True, error_score="raise")
        search.fit(X, y)
    return search


def prepare_target(layer: str, target: str):
    df = load_layer(layer)
    hold_file = DATA_DIR / ("frozen_holdout_primer.csv" if layer == "primer" else "frozen_holdout_topcoat.csv")
    hold_ids = set(pd.read_csv(hold_file, encoding="utf-8-sig")["sample_id"].astype(str))
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].notna()].copy()
    train = df[~df["sample_id"].isin(hold_ids)].reset_index(drop=True)
    hold = df[df["sample_id"].isin(hold_ids)].reset_index(drop=True)
    return train, hold


def aggregate(predictions: pd.DataFrame, fold_metrics: pd.DataFrame):
    keys = ["layer", "target", "target_role", "representation", "model", "scheme"]
    rows = []
    for key, g in predictions.groupby(keys):
        m = metrics(g["y_true"], g["y_pred"])
        fm = fold_metrics
        for col, value in zip(keys, key):
            fm = fm[fm[col] == value]
        row = dict(zip(keys, key))
        row.update({f"pooled_{k}": v for k, v in m.items()})
        row.update(
            {
                "fold_r2_mean": fm["r2"].mean(),
                "fold_r2_std": fm["r2"].std(),
                "fold_rmse_mean": fm["rmse"].mean(),
                "fold_spearman_mean": fm["spearman"].mean(),
                "n_predictions": len(g),
                "n_unique_samples": g["sample_id"].nunique(),
                "n_folds": fm["fold_id"].nunique(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_models(agg: pd.DataFrame):
    strict = agg[agg["scheme"].isin(["family_logo", "level_logo"])].copy()
    keys = ["layer", "target", "target_role", "representation", "model"]
    rank = strict.groupby(keys).agg(
        strict_nrmse=("pooled_nrmse_range", "mean"),
        strict_spearman=("pooled_spearman", "mean"),
        strict_r2=("pooled_r2", "mean"),
        strict_scheme_n=("scheme", "nunique"),
    ).reset_index()
    complexity = {"Dummy": 9, "Ridge": 1, "PLS": 2, "GaussianProcess": 3, "RandomForest": 4, "XGBoost": 5}
    rank["complexity_rank"] = rank["model"].map(complexity).fillna(8)
    rank = rank.sort_values(["layer", "target", "strict_scheme_n", "strict_nrmse", "complexity_rank"], ascending=[True, True, False, True, True])
    rank["selection_rank"] = rank.groupby(["layer", "target"]).cumcount() + 1
    return rank, rank[rank["selection_rank"] == 1].copy()


def plot_validation(agg: pd.DataFrame, selected: pd.DataFrame):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    labels = []
    rows = []
    for _, s in selected.iterrows():
        subset = agg[(agg["layer"] == s["layer"]) & (agg["target"] == s["target"]) & (agg["representation"] == s["representation"]) & (agg["model"] == s["model"])]
        label = f"{s['layer']}\n{s['target'].replace('_pass_h','').replace('_mass_loss_mg','').replace('_mpa_or_grade','')}"
        labels.append(label)
        rows.append({r["scheme"]: r["pooled_r2"] for _, r in subset.iterrows()})
    x = np.arange(len(labels))
    width = 0.24
    colors = {"random_repeated": "#4C78A8", "family_logo": "#F58518", "level_logo": "#54A24B"}
    fig, ax = plt.subplots(figsize=(9.4, 4.5), constrained_layout=True)
    for j, scheme in enumerate(["random_repeated", "family_logo", "level_logo"]):
        vals = [r.get(scheme, np.nan) for r in rows]
        ax.bar(x + (j - 1) * width, vals, width, label=scheme, color=colors[scheme], edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("汇总 R²")
    ax.set_title("随机划分与严格外推验证的性能差异", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.savefig(FIG_DIR / "fig3_validation_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_validation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_predictions, all_fold_metrics = [], []
    specs = model_specs()
    print("MODELS", list(specs))

    for layer, target, role in TARGET_SPECS:
        train, _ = prepare_target(layer, target)
        y = train[target].to_numpy(float)
        reps = ["raw", "physical"] if layer == "primer" else ["raw", "physical", "ilr"]
        schemes = outer_schemes(layer, train)
        print(f"TARGET {layer} {target} n={len(train)} reps={reps}")
        for rep in reps:
            for model_name, (estimator, grid) in specs.items():
                for scheme_name, split_list in schemes.items():
                    for fold_id, (tr, te, inner_groups) in enumerate(split_list):
                        pipe = pipeline(layer, rep, estimator)
                        try:
                            search = fit_search(pipe, grid, train.iloc[tr], y[tr], scheme_name, inner_groups, seed_offset=fold_id)
                            pred = np.asarray(search.predict(train.iloc[te])).ravel()
                            best_params = json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True)
                        except Exception as exc:
                            print("FIT_FAIL", layer, target, rep, model_name, scheme_name, fold_id, repr(exc))
                            continue
                        m = metrics(y[te], pred)
                        fold_row = {
                            "layer": layer, "target": target, "target_role": role,
                            "representation": rep, "model": model_name, "scheme": scheme_name,
                            "fold_id": fold_id, "train_n": len(tr), "test_n": len(te),
                            "best_params": best_params, **m,
                        }
                        all_fold_metrics.append(fold_row)
                        for local_idx, idx in enumerate(te):
                            all_predictions.append(
                                {
                                    "layer": layer, "target": target, "target_role": role,
                                    "representation": rep, "model": model_name, "scheme": scheme_name,
                                    "fold_id": fold_id, "sample_id": train.iloc[idx]["sample_id"],
                                    "formulation_family": train.iloc[idx]["formulation_family"],
                                    "y_true": float(y[idx]), "y_pred": float(pred[local_idx]),
                                }
                            )

    predictions = pd.DataFrame(all_predictions)
    fold_metrics = pd.DataFrame(all_fold_metrics)
    predictions.to_csv(RESULT_DIR / "nested_cv_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(RESULT_DIR / "nested_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    agg = aggregate(predictions, fold_metrics)
    agg.to_csv(RESULT_DIR / "nested_cv_summary.csv", index=False, encoding="utf-8-sig")
    ranking, selected = select_models(agg)
    ranking.to_csv(RESULT_DIR / "model_selection_ranking.csv", index=False, encoding="utf-8-sig")

    final_rows, hold_rows = [], []
    for _, chosen in selected.iterrows():
        layer, target = chosen["layer"], chosen["target"]
        role, rep, model_name = chosen["target_role"], chosen["representation"], chosen["model"]
        train, hold = prepare_target(layer, target)
        estimator, grid = specs[model_name]
        final_search = fit_search(pipeline(layer, rep, estimator), grid, train, train[target].to_numpy(float), "random_repeated", None)
        model_path = MODEL_DIR / f"final_{layer}_{target}.joblib"
        joblib.dump(final_search.best_estimator_, model_path)
        feature_names = final_search.best_estimator_.named_steps["repr"].get_feature_names_out().tolist()
        (DATA_DIR / f"final_features_{layer}_{target}.txt").write_text("\n".join(feature_names), encoding="utf-8")
        final_rows.append(
            {
                **chosen.to_dict(),
                "training_n": len(train), "holdout_n": len(hold),
                "best_params": json.dumps(final_search.best_params_, ensure_ascii=False, sort_keys=True),
                "model_path": str(model_path), "feature_n": len(feature_names),
            }
        )
        hp = np.asarray(final_search.predict(hold)).ravel()
        for i, pred in enumerate(hp):
            hold_rows.append(
                {
                    "layer": layer, "target": target, "target_role": role,
                    "sample_id": hold.iloc[i]["sample_id"],
                    "formulation_family": hold.iloc[i]["formulation_family"],
                    "representation": rep, "model": model_name,
                    "y_true": float(hold.iloc[i][target]), "y_pred": float(pred),
                }
            )

    final_selection = pd.DataFrame(final_rows)
    final_selection.to_csv(RESULT_DIR / "final_model_selection.csv", index=False, encoding="utf-8-sig")
    holdout = pd.DataFrame(hold_rows)
    holdout.to_csv(RESULT_DIR / "holdout_point_predictions.csv", index=False, encoding="utf-8-sig")
    plot_validation(agg, selected)

    print("STRICT_MODELING_OK")
    print(final_selection[["layer", "target", "target_role", "representation", "model", "strict_nrmse", "strict_spearman", "strict_r2"]].to_string(index=False))
    print("HOLDOUT_POINT_METRICS")
    for key, g in holdout.groupby(["layer", "target"]):
        print(key, metrics(g["y_true"], g["y_pred"]))


if __name__ == "__main__":
    main()
