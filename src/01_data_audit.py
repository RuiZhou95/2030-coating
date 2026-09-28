#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only audit of the independent primer and topcoat workbooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from project_paths import PRIMER_FILE, PROJECT_ROOT as PROJECT, RAW_DATA_DIR as RAW_DIR, SHEET_NAME, TOPCOAT_FILE

DATA_DIR = PROJECT / "data"
RESULT_DIR = PROJECT / "results"
FIG_DIR = PROJECT / "figure"

FILES = {
    "底漆": RAW_DIR / PRIMER_FILE,
    "面漆": RAW_DIR / TOPCOAT_FILE,
}
TARGETS = ["salt_spray_pass_h", "adhesion_mpa_or_grade", "wear_mass_loss_mg"]
TARGET_ALIASES = {"底漆": ["salt_spray_time_h"], "面漆": ["salt_spray_time_h"]}
PRIMER_COMPONENTS = ["铬酸锶", "磷酸锌"]
TOPCOAT_COMPONENTS = [
    "聚氨酯树脂A14含量", "氟树脂含量", "碳化硅", "氮化硼",
    "润滑粉", "分散剂", "防沉剂", "消泡剂",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_column_name(value) -> str:
    return str(value).replace("\n", "").strip()


def load_layer(layer: str, path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        header=1,
        skiprows=[2, 3, 4, 5, 6, 7],
    )
    df.columns = [clean_column_name(c) for c in df.columns]
    if layer == "面漆":
        raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=None, nrows=3)
        rename = {}
        for i in range(8, min(16, raw.shape[1])):
            name = clean_column_name(raw.iloc[2, i])
            if name and name.lower() != "nan":
                rename[f"Unnamed: {i}"] = name
        df = df.rename(columns=rename)
    aliases = TARGET_ALIASES.get(layer, [])
    df = df.drop(columns=aliases, errors="ignore")
    if any(alias in df.columns for alias in aliases):
        raise AssertionError(f"{layer}目标重复字段未删除: {aliases}")
    if "sample_id" not in df.columns:
        raise ValueError(f"{layer}缺少 sample_id 列")
    df = df.dropna(subset=["sample_id"]).copy()
    df = df[~df["sample_id"].astype(str).str.contains("第10行", na=False)].copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df.insert(0, "layer", layer)
    return df.reset_index(drop=True)


def dictionary_rows(layer: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        numeric = pd.to_numeric(s, errors="coerce")
        examples = " | ".join(map(str, non_null.astype(str).drop_duplicates().head(3)))
        row = {
            "layer": layer,
            "column": col,
            "dtype": str(s.dtype),
            "non_null_n": int(s.notna().sum()),
            "missing_n": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 3),
            "unique_n": int(non_null.nunique()),
            "examples": examples,
            "numeric_n": int(numeric.notna().sum()),
            "min": float(numeric.min()) if numeric.notna().any() else np.nan,
            "max": float(numeric.max()) if numeric.notna().any() else np.nan,
            "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
            "std": float(numeric.std()) if numeric.notna().sum() > 1 else np.nan,
        }
        rows.append(row)
    return rows


def present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def numeric_design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in present(df, cols):
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def boundary_flag(x: pd.DataFrame) -> pd.Series:
    if x.empty:
        return pd.Series(False, index=x.index)
    flags = pd.DataFrame(False, index=x.index, columns=x.columns)
    for c in x.columns:
        s = x[c]
        if s.notna().sum() and s.nunique(dropna=True) > 1:
            flags[c] = np.isclose(s, s.min(), equal_nan=False) | np.isclose(
                s, s.max(), equal_nan=False
            )
    return flags.sum(axis=1) >= max(1, int(np.ceil(x.shape[1] / 3)))


def make_groups(primer: pd.DataFrame, topcoat: pd.DataFrame) -> pd.DataFrame:
    out = []
    p = numeric_design(primer, PRIMER_COMPONENTS)
    sr = p.get("铬酸锶", pd.Series(0.0, index=primer.index)).fillna(0)
    zp = p.get("磷酸锌", pd.Series(0.0, index=primer.index)).fillna(0)
    p_family = np.select(
        [(sr <= 0) & (zp <= 0), (sr > 0) & (zp <= 0), (sr <= 0) & (zp > 0)],
        ["无显式抑制剂", "铬酸锶单一", "磷酸锌单一"],
        default="混合抑制剂",
    )
    for i, row in primer.iterrows():
        item = {
            "layer": "底漆",
            "sample_id": row["sample_id"],
            "formulation_family": p_family[i],
            "is_design_boundary": bool(boundary_flag(p).iloc[i]),
        }
        for t in TARGETS:
            item[t] = pd.to_numeric(row.get(t), errors="coerce")
        out.append(item)

    m = numeric_design(topcoat, TOPCOAT_COMPONENTS)
    fluor = m.get("氟树脂含量", pd.Series(0.0, index=topcoat.index)).fillna(0)
    sic = m.get("碳化硅", pd.Series(0.0, index=topcoat.index)).fillna(0)
    bn = m.get("氮化硼", pd.Series(0.0, index=topcoat.index)).fillna(0)
    binder = np.where(fluor <= 0, "聚氨酯基", "含氟改性")
    ceramic = np.select(
        [sic > 1.25 * bn, bn > 1.25 * sic],
        ["SiC主导", "BN主导"],
        default="SiC-BN平衡",
    )
    for i, row in topcoat.iterrows():
        item = {
            "layer": "面漆",
            "sample_id": row["sample_id"],
            "formulation_family": f"{binder[i]}_{ceramic[i]}",
            "is_design_boundary": bool(boundary_flag(m).iloc[i]),
        }
        for t in TARGETS:
            item[t] = pd.to_numeric(row.get(t), errors="coerce")
        out.append(item)
    groups = pd.DataFrame(out)
    counts = groups.groupby(["layer", "formulation_family"])["sample_id"].transform("count")
    groups["family_n"] = counts.astype(int)
    return groups


def audit_layer(layer: str, df: pd.DataFrame, design_cols: list[str]) -> dict:
    d = numeric_design(df, design_cols)
    valid_design = d.dropna(axis=1, how="all")
    dup_design = int(valid_design.duplicated(keep=False).sum()) if not valid_design.empty else 0
    all_null = [c for c in df.columns if df[c].notna().sum() == 0]
    constant = [c for c in df.columns if df[c].dropna().nunique() <= 1]
    summary = {
        "layer": layer,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_sample_ids": int(df["sample_id"].duplicated(keep=False).sum()),
        "duplicate_design_rows": dup_design,
        "all_null_columns_n": len(all_null),
        "constant_columns_n": len(constant),
        "missing_cells_n": int(df.isna().sum().sum()),
        "all_null_columns": " | ".join(all_null),
        "constant_columns": " | ".join(constant),
    }
    if layer == "面漆":
        comps = numeric_design(df, TOPCOAT_COMPONENTS)
        comp_sum = comps.sum(axis=1, min_count=1)
        summary.update(
            {
                "composition_sum_mean": float(comp_sum.mean()),
                "composition_sum_min": float(comp_sum.min()),
                "composition_sum_max": float(comp_sum.max()),
                "composition_sum_within_99_101_pct": float(
                    ((comp_sum >= 99) & (comp_sum <= 101)).mean() * 100
                ),
                "zero_component_pct": float((comps == 0).mean().mean() * 100),
            }
        )
    return summary


def target_stats(layer: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for t in TARGETS:
        if t not in df.columns:
            continue
        y = pd.to_numeric(df[t], errors="coerce")
        rows.append(
            {
                "layer": layer,
                "target": t,
                "n": int(y.notna().sum()),
                "missing_n": int(y.isna().sum()),
                "unique_n": int(y.nunique(dropna=True)),
                "min": float(y.min()) if y.notna().any() else np.nan,
                "q25": float(y.quantile(0.25)) if y.notna().any() else np.nan,
                "median": float(y.median()) if y.notna().any() else np.nan,
                "mean": float(y.mean()) if y.notna().any() else np.nan,
                "q75": float(y.quantile(0.75)) if y.notna().any() else np.nan,
                "max": float(y.max()) if y.notna().any() else np.nan,
                "std": float(y.std()) if y.notna().sum() > 1 else np.nan,
            }
        )
    return rows


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "savefig.dpi": 600,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def make_figure(primer: pd.DataFrame, topcoat: pd.DataFrame):
    style()
    blue, red, gold, gray = "#2F6690", "#C8553D", "#D9A441", "#6B7280"
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.8), constrained_layout=True)

    def distribution(ax, series, color, title, xlabel):
        y = pd.to_numeric(series, errors="coerce").dropna()
        bins = min(14, max(7, int(np.sqrt(len(y)))))
        counts, _, _ = ax.hist(y, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.6)
        median = y.median()
        ax.axvline(median, color="#222222", linestyle="--", linewidth=1.8)
        ymax = max(float(np.nanmax(counts)), 1.0)
        ax.set_ylim(0, ymax * 1.18)
        label_x = median + 0.025 * max(float(y.max() - y.min()), 1.0)
        if abs(median) >= 100:
            median_text = f"{float(f'{median:.3g}'):.0f}"
        else:
            median_text = f"{median:.2f}".rstrip("0").rstrip(".")
        ax.text(
            label_x, ymax * 1.08, f"Median = {median_text}",
            ha="left", va="bottom", fontsize=14, color="#222222",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title, loc="left", fontweight="bold")

    distribution(axes[0, 0], primer["salt_spray_pass_h"], blue, "a  Primer salt-spray lifetime", "Salt-spray failure time (h)")
    distribution(axes[0, 1], primer["adhesion_mpa_or_grade"], red, "b  Primer adhesion strength", "Adhesion strength (MPa)")

    ax = axes[0, 2]
    sr = pd.to_numeric(primer.get("铬酸锶"), errors="coerce")
    zp = pd.to_numeric(primer.get("磷酸锌"), errors="coerce")
    salt = pd.to_numeric(primer.get("salt_spray_pass_h"), errors="coerce")
    sc = ax.scatter(sr, zp, c=salt, cmap="viridis", s=126, edgecolor="white", linewidth=0.6)
    cb = fig.colorbar(sc, ax=ax, label="Salt-spray lifetime (h)")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.set_xlabel("Strontium chromate (wt.%)")
    ax.set_ylabel("Zinc phosphate (wt.%)")
    ax.set_ylim(-2, 10)
    ax.set_title("c  Primer inhibitor design space", loc="left", fontweight="bold")

    distribution(axes[1, 0], topcoat["salt_spray_pass_h"], gold, "d  Topcoat salt-spray lifetime", "Salt-spray failure time (h)")
    distribution(axes[1, 1], topcoat["wear_mass_loss_mg"], blue, "e  Topcoat wear loss", "Wear mass loss (mg)")

    ax = axes[1, 2]
    sic = pd.to_numeric(topcoat.get("碳化硅"), errors="coerce")
    bn = pd.to_numeric(topcoat.get("氮化硼"), errors="coerce")
    salt = pd.to_numeric(topcoat.get("salt_spray_pass_h"), errors="coerce")
    fluor = pd.to_numeric(topcoat.get("氟树脂含量"), errors="coerce").fillna(0)
    sizes = 50 + 6.0 * fluor.clip(lower=0)
    sc = ax.scatter(
        sic, bn, c=salt, s=sizes, cmap="viridis", norm=plt.Normalize(500, 3200),
        alpha=0.95, edgecolor="#1F2933", linewidth=0.8,
    )
    cb = fig.colorbar(sc, ax=ax, label="Salt-spray lifetime (h)")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.set_xlabel("Silicon carbide (wt.%)")
    ax.set_ylabel("Boron nitride (wt.%)")
    ax.set_title(
        "f  Topcoat ceramic-filler design space\nMarker size denotes fluororesin content",
        loc="left", fontweight="bold", fontsize=18,
    )

    fig.savefig(FIG_DIR / "fig1_data_space.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_data_space.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_data_space.svg", bbox_inches="tight")
    plt.close(fig)


def fmt(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


def write_report(layer_summaries: pd.DataFrame, targets: pd.DataFrame, groups: pd.DataFrame):
    lines = [
        "# 底漆与面漆数据谱系和质量审计",
        "",
        "本报告由只读审计脚本生成。原始工作簿未被修改或复制。底漆与面漆为独立数据域，不支持体系级协同结论。",
        "",
        "## 1. 数据规模与结构",
        "",
    ]
    for _, r in layer_summaries.iterrows():
        lines.extend(
            [
                f"### {r['layer']}",
                "",
                f"- 样本数：{int(r['n_rows'])}",
                f"- 字段数：{int(r['n_columns'])}",
                f"- 重复样本编号数：{int(r['duplicate_sample_ids'])}",
                f"- 重复设计行数：{int(r['duplicate_design_rows'])}",
                f"- 全空字段数：{int(r['all_null_columns_n'])}",
                f"- 恒定字段数：{int(r['constant_columns_n'])}",
            ]
        )
        if r["layer"] == "面漆":
            lines.extend(
                [
                    f"- 八个显式组分之和：均值 {fmt(r.get('composition_sum_mean'))}，范围 {fmt(r.get('composition_sum_min'))}–{fmt(r.get('composition_sum_max'))}",
                    f"- 组分和位于 99–101 的样本比例：{fmt(r.get('composition_sum_within_99_101_pct'))}%",
                    f"- 组分零值比例：{fmt(r.get('zero_component_pct'))}%",
                ]
            )
        lines.append("")

    lines.extend(["## 2. 性能目标", "", "| 层级 | 目标 | n | 缺失 | 唯一值 | 最小 | 中位数 | 最大 |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for _, r in targets.iterrows():
        lines.append(
            f"| {r['layer']} | {r['target']} | {int(r['n'])} | {int(r['missing_n'])} | {int(r['unique_n'])} | {fmt(r['min'])} | {fmt(r['median'])} | {fmt(r['max'])} |"
        )

    lines.extend(["", "## 3. 可解释配方族", "", "| 层级 | 配方族 | 样本数 |", "|---|---|---:|"])
    fam = groups.groupby(["layer", "formulation_family"]).size().reset_index(name="n")
    for _, r in fam.iterrows():
        lines.append(f"| {r['layer']} | {r['formulation_family']} | {int(r['n'])} |")

    lines.extend(
        [
            "",
            "## 4. 对正式建模的约束",
            "",
            "1. 随机划分只能作为历史对照，不能证明对新配方族或新组分水平的外推能力。",
            "2. 面漆显式组分按组成数据处理；零值替换、CLR/ILR和缩放必须在交叉验证折内完成。",
            "3. 底漆和面漆分别建模，不拼接特征空间，不生成配套体系评分。",
            "4. 正式建模前冻结少量高性能样本和中性能对照，并保存不可变划分清单。",
            "5. 任何机理表述均需同时满足统计稳定性、物理合理性和不同划分下方向一致性。",
            "",
            "## 5. 产物",
            "",
            "- `data/data_dictionary.csv`：逐字段数据字典。",
            "- `data/data_quality_summary.csv`：层级审计汇总。",
            "- `data/target_statistics.csv`：目标统计。",
            "- `data/sample_groups.csv`：可解释配方族与边界标记。",
            "- `data/source_manifest.json`：原始文件路径、大小和 SHA256。",
            "- `data/removed_target_aliases.csv`：从分析数据中删除的目标重复字段。",
            "- `figure/fig1_data_space.*`：设计空间与目标分布。",
        ]
    )
    (RESULT_DIR / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    for directory in [DATA_DIR, RESULT_DIR, FIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for path in FILES.values():
        if not path.exists():
            raise FileNotFoundError(path)

    primer = load_layer("底漆", FILES["底漆"])
    topcoat = load_layer("面漆", FILES["面漆"])
    if "salt_spray_time_h" in primer.columns or "salt_spray_time_h" in topcoat.columns:
        raise AssertionError("salt_spray_time_h must not enter cleaned analysis data")

    removed_aliases = pd.DataFrame(
        [
            {
                "layer": "底漆",
                "column": "salt_spray_time_h",
                "duplicate_of": "salt_spray_pass_h",
                "action": "dropped_during_import",
                "reason": "target alias/legacy field; retained only in immutable source workbook",
            },
            {
                "layer": "面漆",
                "column": "salt_spray_time_h",
                "duplicate_of": "salt_spray_pass_h",
                "action": "dropped_during_import",
                "reason": "exact target alias; retained only in immutable source workbook",
            }
        ]
    )
    removed_aliases.to_csv(
        DATA_DIR / "removed_target_aliases.csv", index=False, encoding="utf-8-sig"
    )

    dictionary = pd.DataFrame(
        dictionary_rows("底漆", primer) + dictionary_rows("面漆", topcoat)
    )
    dictionary.to_csv(DATA_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")

    layer_summaries = pd.DataFrame(
        [
            audit_layer("底漆", primer, PRIMER_COMPONENTS + ["coating_cure_temp_c", "coating_cure_time_h", "surface_roughness_ra_um"]),
            audit_layer("面漆", topcoat, TOPCOAT_COMPONENTS),
        ]
    )
    layer_summaries.to_csv(DATA_DIR / "data_quality_summary.csv", index=False, encoding="utf-8-sig")

    targets = pd.DataFrame(target_stats("底漆", primer) + target_stats("面漆", topcoat))
    targets.to_csv(DATA_DIR / "target_statistics.csv", index=False, encoding="utf-8-sig")

    groups = make_groups(primer, topcoat)
    groups.to_csv(DATA_DIR / "sample_groups.csv", index=False, encoding="utf-8-sig")

    manifest = {
        layer: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "sheet": "coating_samples",
        }
        for layer, path in FILES.items()
    }
    (DATA_DIR / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    make_figure(primer, topcoat)
    write_report(layer_summaries, targets, groups)

    print(f"AUDIT_OK primer={len(primer)} topcoat={len(topcoat)}")
    print(layer_summaries.to_string(index=False))
    print(targets.to_string(index=False))
    print(groups.groupby(["layer", "formulation_family"]).size())


if __name__ == "__main__":
    main()
