#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose structured design/leakage and freeze retrospective holdouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from project_paths import PRIMER_FILE, PROJECT_ROOT as PROJECT, RAW_DATA_DIR as RAW_DIR, SHEET_NAME, TOPCOAT_FILE

DATA_DIR = PROJECT / "data"
RESULT_DIR = PROJECT / "results"

FILES = {
    "底漆": RAW_DIR / PRIMER_FILE,
    "面漆": RAW_DIR / TOPCOAT_FILE,
}
TARGETS = ["salt_spray_pass_h", "adhesion_mpa_or_grade", "wear_mass_loss_mg"]
POST_OUTCOME_COLUMNS = {
    "salt_spray_time_h", "adhesion_mpa_or_grade", "wear_mass_loss_mg",
    "pressure_resistance_result", "applied_stress_mpa", "static_pressure_hold_h",
    "dynamic_pressure_cycle_count", "erosion_duration_h", "blistering_grade",
    "peeling_or_flaking_grade",
}
PRIMER_DESIGN = [
    "铬酸锶", "磷酸锌", "coating_curing_agent_type",
    "coating_cure_temp_c", "coating_cure_time_h", "surface_roughness_ra_um",
]
TOPCOAT_COMPONENTS = [
    "聚氨酯树脂A14含量", "氟树脂含量", "碳化硅", "氮化硼",
    "润滑粉", "分散剂", "防沉剂", "消泡剂",
]


def clean_name(value) -> str:
    return str(value).replace("\n", "").strip()


def load_layer(layer: str, path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=1, skiprows=[2, 3, 4, 5, 6, 7])
    df.columns = [clean_name(c) for c in df.columns]
    if layer == "面漆":
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
    df.insert(0, "layer", layer)
    return df.reset_index(drop=True)


def add_groups(primer: pd.DataFrame, topcoat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pd.read_csv(DATA_DIR / "sample_groups.csv", encoding="utf-8-sig")
    p = primer.merge(groups[["layer", "sample_id", "formulation_family", "is_design_boundary"]], on=["layer", "sample_id"], how="left")
    m = topcoat.merge(groups[["layer", "sample_id", "formulation_family", "is_design_boundary"]], on=["layer", "sample_id"], how="left")
    return p, m


def factor_levels(layer: str, df: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        values = [str(v) for v in sorted(s.unique(), key=lambda x: str(x))]
        rows.append(
            {
                "layer": layer,
                "feature": c,
                "unique_n": int(s.nunique()),
                "values": " | ".join(values[:50]),
                "truncated": len(values) > 50,
            }
        )
    return rows


def numeric_copy(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def leakage_scan(layer: str, df: pd.DataFrame) -> pd.DataFrame:
    ignored = {"sample_id"}
    rows = []
    numeric = {}
    for c in df.columns:
        if c in ignored:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == len(df) and s.nunique() > 1:
            numeric[c] = s.astype(float)
    for target in TARGETS:
        if target not in numeric:
            continue
        y = numeric[target]
        for feature, x in numeric.items():
            if feature == target:
                continue
            exact_equal = bool(np.allclose(x, y, rtol=0, atol=1e-12))
            rho, _ = spearmanr(x, y)
            if exact_equal or (np.isfinite(rho) and abs(rho) >= 0.90):
                exclude = bool(exact_equal or feature in POST_OUTCOME_COLUMNS or feature in TARGETS)
                risk_type = "明确泄漏/其他结果列" if exclude else "强设计关联，保留并做稳定性审计"
                rows.append(
                    {
                        "layer": layer,
                        "target": target,
                        "candidate_column": feature,
                        "exact_equal": exact_equal,
                        "spearman_rho": float(rho),
                        "risk_type": risk_type,
                        "exclude_from_model": exclude,
                    }
                )
    return pd.DataFrame(rows)


def target_frequencies(layer: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce").dropna()
        for value, n in y.value_counts().sort_index().items():
            rows.append({"layer": layer, "target": target, "value": value, "n": int(n)})
    return pd.DataFrame(rows)


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """All columns are maximized."""
    n = len(values)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.any(
            np.all(values >= values[i], axis=1)
            & np.any(values > values[i], axis=1)
        )
        keep[i] = not dominated
    return keep


def percentile_score(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    pct = s.rank(method="average", pct=True)
    return pct if higher_is_better else 1.0 - pct + 1.0 / len(s)


def diversified_select(df: pd.DataFrame, score_col: str, pool_mask: pd.Series, n: int) -> list[int]:
    pool = df.loc[pool_mask].sort_values(score_col, ascending=False)
    selected = []
    seen = set()
    for idx, row in pool.iterrows():
        fam = row["formulation_family"]
        if fam not in seen:
            selected.append(idx)
            seen.add(fam)
        if len(selected) >= n:
            return selected
    for idx in pool.index:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= n:
            break
    return selected


def freeze_layer(layer: str, df: pd.DataFrame, high_n: int = 6, control_n: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if layer == "底漆":
        salt = pd.to_numeric(work["salt_spray_pass_h"], errors="coerce")
        adhesion = pd.to_numeric(work["adhesion_mpa_or_grade"], errors="coerce")
        salt_pct = percentile_score(salt, True)
        secondary_pct = percentile_score(adhesion, True)
        work["selection_score"] = (salt_pct + secondary_pct) / 2
        objectives = np.column_stack([salt, adhesion])
        design_cols = [c for c in PRIMER_DESIGN if c in work.columns]
    else:
        salt = pd.to_numeric(work["salt_spray_pass_h"], errors="coerce")
        wear = pd.to_numeric(work["wear_mass_loss_mg"], errors="coerce")
        salt_pct = percentile_score(salt, True)
        secondary_pct = percentile_score(wear, False)
        work["selection_score"] = (salt_pct + secondary_pct) / 2
        objectives = np.column_stack([salt, -wear])
        design_cols = [c for c in TOPCOAT_COMPONENTS if c in work.columns]
    work["is_pareto_observed"] = pareto_mask(objectives)
    work["objective_balance_distance"] = (salt_pct - 0.5).abs() + (secondary_pct - 0.5).abs()

    high_pool = work["is_pareto_observed"] | (work["selection_score"] >= work["selection_score"].quantile(0.85))
    high_idx = diversified_select(work, "selection_score", high_pool, high_n)

    remaining = work.drop(index=high_idx).copy()
    remaining["mid_distance"] = remaining["objective_balance_distance"]
    mid_pool = remaining.sort_values("mid_distance")
    control_idx = []
    seen = {work.loc[i, "formulation_family"] for i in high_idx}
    for idx, row in mid_pool.iterrows():
        if row["formulation_family"] not in seen or len(control_idx) + len(seen) >= work["formulation_family"].nunique():
            control_idx.append(idx)
            seen.add(row["formulation_family"])
        if len(control_idx) >= control_n:
            break
    if len(control_idx) < control_n:
        for idx in remaining.sort_values("mid_distance").index:
            if idx not in control_idx:
                control_idx.append(idx)
            if len(control_idx) >= control_n:
                break

    frozen = work.loc[high_idx + control_idx].copy()
    frozen["holdout_role"] = ["高性能/Pareto"] * len(high_idx) + ["中性能对照"] * len(control_idx)
    frozen["frozen_before_modeling"] = True
    frozen["selection_policy_version"] = "v2_20260823"
    keep_cols = [
        "layer", "sample_id", "formulation_family", "is_design_boundary",
        "holdout_role", "selection_score", "objective_balance_distance", "is_pareto_observed",
    ] + [c for c in TARGETS if c in frozen.columns] + design_cols
    frozen = frozen[keep_cols].sort_values(["holdout_role", "selection_score"], ascending=[True, False])
    training = work[~work["sample_id"].isin(frozen["sample_id"])].copy()
    return frozen.reset_index(drop=True), training.reset_index(drop=True)


def digest_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def write_report(
    primer: pd.DataFrame,
    topcoat: pd.DataFrame,
    levels: pd.DataFrame,
    leakage: pd.DataFrame,
    p_hold: pd.DataFrame,
    m_hold: pd.DataFrame,
):
    p_factors = ["铬酸锶", "磷酸锌", "coating_curing_agent_type"]
    p = primer.copy()
    p["cure_profile"] = p["coating_cure_temp_c"].astype(str) + "C_" + p["coating_cure_time_h"].astype(str) + "h"
    observed = p[p_factors + ["cure_profile"]].drop_duplicates().shape[0]
    expected = int(np.prod([p[c].nunique() for c in p_factors + ["cure_profile"]]))
    cure_pairs = p[["coating_cure_temp_c", "coating_cure_time_h"]].drop_duplicates()

    m = numeric_copy(topcoat, TOPCOAT_COMPONENTS)
    grouped = pd.DataFrame(
        {
            "binder_total": m["聚氨酯树脂A14含量"] + m["氟树脂含量"],
            "ceramic_total": m["碳化硅"] + m["氮化硼"],
            "lubricant": m["润滑粉"],
            "additives_total": m["分散剂"] + m["防沉剂"] + m["消泡剂"],
        }
    )

    lines = [
        "# 结构化设计、泄漏风险与冻结验证集",
        "",
        "## 1. 底漆设计结构",
        "",
        f"底漆的铬酸锶、磷酸锌、固化剂类型和固化制度共有 {observed} 个不同组合；各因子水平笛卡尔积为 {expected}。",
        f"固化温度—时间只有 {len(cure_pairs)} 个组合，说明二者构成绑定的固化制度，不应作为两个完全独立因子解释。",
        "这表明底漆接近完整规则因子设计。随机划分会把相邻因子组合同时放入训练和测试集，不能代表新水平或新配方族外推。",
        "",
        "## 2. 面漆组成结构",
        "",
        f"面漆显式组分严格闭合。树脂相总量范围 {grouped['binder_total'].min():.2f}–{grouped['binder_total'].max():.2f}，陶瓷填料相范围 {grouped['ceramic_total'].min():.2f}–{grouped['ceramic_total'].max():.2f}。",
        f"盐雾目标只有 {pd.to_numeric(topcoat['salt_spray_pass_h'], errors='coerce').nunique()} 个离散失效时间水平，磨耗目标有 {pd.to_numeric(topcoat['wear_mass_loss_mg'], errors='coerce').nunique()} 个水平。",
        "因此需要报告等级相关、区间覆盖和组分水平留出，不能只报告随机划分 R²。",
        "",
        "## 3. 泄漏候选字段",
        "",
        "以下字段与目标完全相等或 Spearman |rho|>=0.90。只有明确泄漏、测试后结果列或其他性能目标需要排除；配方成分的强关联保留，但必须接受折间稳定性和组成共线审计：",
        "",
        "| 层级 | 目标 | 候选字段 | 完全相等 | Spearman rho | 风险类型 | 排除 |",
        "|---|---|---|---|---:|---|---|",
    ]
    for _, r in leakage.iterrows():
        lines.append(f"| {r['layer']} | {r['target']} | {r['candidate_column']} | {r['exact_equal']} | {r['spearman_rho']:.3f} | {r['risk_type']} | {r['exclude_from_model']} |")

    lines.extend(
        [
            "",
            "## 4. 冻结验证集",
            "",
            f"- 底漆冻结 {len(p_hold)} 条：{(p_hold['holdout_role']=='高性能/Pareto').sum()} 条高性能/Pareto，{(p_hold['holdout_role']=='中性能对照').sum()} 条中性能对照。",
            f"- 面漆冻结 {len(m_hold)} 条：{(m_hold['holdout_role']=='高性能/Pareto').sum()} 条高性能/Pareto，{(m_hold['holdout_role']=='中性能对照').sum()} 条中性能对照。",
            "- 冻结集不参与特征选择、表示选择、调参、模型筛选、SHAP或Pareto规则确定。",
            "- 冻结集仅用于最终排名恢复、预测区间覆盖、高性能识别和外推偏差评估。",
            "",
            "## 5. 正式验证要求",
            "",
            "1. 底漆采用因子水平留出、固化剂/抑制剂配方族留出与嵌套交叉验证。",
            "2. 面漆采用配方族留出、关键组分分位水平留出与嵌套交叉验证。",
            "3. 面漆组成变换的零值替换必须在每个训练折内拟合。",
            "4. 与随机划分结果并列展示，量化乐观偏差。",
        ]
    )
    (RESULT_DIR / "design_structure_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    primer = load_layer("底漆", FILES["底漆"])
    topcoat = load_layer("面漆", FILES["面漆"])
    primer, topcoat = add_groups(primer, topcoat)

    levels = pd.DataFrame(
        factor_levels("底漆", primer, PRIMER_DESIGN)
        + factor_levels("面漆", topcoat, TOPCOAT_COMPONENTS)
    )
    levels.to_csv(DATA_DIR / "design_factor_levels.csv", index=False, encoding="utf-8-sig")

    frequencies = pd.concat(
        [target_frequencies("底漆", primer), target_frequencies("面漆", topcoat)],
        ignore_index=True,
    )
    frequencies.to_csv(DATA_DIR / "target_value_frequencies.csv", index=False, encoding="utf-8-sig")

    leakage = pd.concat(
        [leakage_scan("底漆", primer), leakage_scan("面漆", topcoat)],
        ignore_index=True,
    )
    leakage.to_csv(DATA_DIR / "leakage_candidates.csv", index=False, encoding="utf-8-sig")

    p_corr = numeric_copy(primer, ["铬酸锶", "磷酸锌", "coating_cure_temp_c", "coating_cure_time_h", "surface_roughness_ra_um"] + TARGETS).corr(method="spearman")
    p_corr.to_csv(DATA_DIR / "correlation_primer_spearman.csv", encoding="utf-8-sig")
    m_corr = numeric_copy(topcoat, TOPCOAT_COMPONENTS + ["salt_spray_pass_h", "wear_mass_loss_mg"]).corr(method="spearman")
    m_corr.to_csv(DATA_DIR / "correlation_topcoat_spearman.csv", encoding="utf-8-sig")

    p_hold, p_train = freeze_layer("底漆", primer)
    m_hold, m_train = freeze_layer("面漆", topcoat)
    p_hold.to_csv(DATA_DIR / "frozen_holdout_primer.csv", index=False, encoding="utf-8-sig")
    m_hold.to_csv(DATA_DIR / "frozen_holdout_topcoat.csv", index=False, encoding="utf-8-sig")
    p_train[["sample_id"]].to_csv(DATA_DIR / "training_ids_primer.csv", index=False, encoding="utf-8-sig")
    m_train[["sample_id"]].to_csv(DATA_DIR / "training_ids_topcoat.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "policy_version": "v2_20260823",
        "frozen_before_modeling": True,
        "primer": {
            "holdout_n": len(p_hold),
            "training_n": len(p_train),
            "holdout_id_sha256": digest_ids(p_hold["sample_id"].tolist()),
            "training_id_sha256": digest_ids(p_train["sample_id"].tolist()),
        },
        "topcoat": {
            "holdout_n": len(m_hold),
            "training_n": len(m_train),
            "holdout_id_sha256": digest_ids(m_hold["sample_id"].tolist()),
            "training_id_sha256": digest_ids(m_train["sample_id"].tolist()),
        },
        "prohibited_uses": [
            "feature_selection", "representation_selection", "hyperparameter_search",
            "model_selection", "shap_fitting", "pareto_rule_definition",
        ],
    }
    (DATA_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_report(primer, topcoat, levels, leakage, p_hold, m_hold)
    print("DESIGN_AUDIT_OK")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("LEAKAGE_CANDIDATES")
    print(leakage.to_string(index=False))
    print("PRIMER_HOLDOUT")
    print(p_hold.to_string(index=False))
    print("TOPCOAT_HOLDOUT")
    print(m_hold.to_string(index=False))


if __name__ == "__main__":
    main()
