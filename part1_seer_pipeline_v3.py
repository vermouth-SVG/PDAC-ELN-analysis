"""
Part 1 submission-ready SEER ELN/POS analysis pipeline.

Outputs:
- outputs_part1_seer_analysis_v3/figures/Figure_1_SEER_ELN_Overview.(png|pdf|tiff)
- outputs_part1_seer_analysis_v3/figures/Figure_2_SEER_ELN_Modeling.(png|pdf|tiff)
- outputs_part1_seer_analysis_v3/figures/Figure_3_SEER_POS_Analysis.(png|pdf|tiff)
- outputs_part1_seer_analysis_v3/figures/Supplementary_Figure_S1_SEER_Model_Performance.(png|pdf|tiff)
- outputs_part1_seer_analysis_v3/figures/Supplementary_Figure_S2_SEER_ELN_By_Stage.(png|pdf|tiff)
- outputs_part1_seer_analysis_v3/figures/Supplementary_Figure_S3_SEER_RMST_OW.(png|pdf|tiff)
- outputs_part1_seer_analysis_v3/tables/Part1_SEER_Tables.xlsx
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from patsy import build_design_matrices, dmatrix
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class AnalysisConfig:
    excel_path: str = "Pancreatic_Body_Tail_ALL.xlsx"
    sheet_name: str = "export"
    outdir: str = "outputs_part1_seer_analysis_v3"
    random_state: int = 42
    fig_dpi: int = 300
    calibration_times: Tuple[int, int, int] = (12, 36, 60)
    km_risk_times: Tuple[int, ...] = (0, 12, 24, 36, 48, 60)
    km_truncation_fraction: float = 1 / 3
    threshold_candidates: Tuple[int, ...] = tuple(range(6, 36))
    changepoint_candidates: Tuple[int, ...] = tuple(range(6, 36))
    eln_bins_detailed: Tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40, 60, 9999)
    eln_bin_labels_detailed: Tuple[str, ...] = ("1-5", "6-10", "11-15", "16-20", "21-30", "31-40", "41-60", "61+")
    eln_bins_plot: Tuple[int, ...] = (0, 10, 20, 30, 9999)
    eln_bin_labels_plot: Tuple[str, ...] = ("1-10", "11-20", "21-30", "31+")
    primary_cut: int = 10
    strategy_cuts: Tuple[int, ...] = (12, 14, 21)
    strategy_methods: Tuple[str, ...] = ("OW", "IPTW")
    cindex_bootstrap_n: int = 200
    changepoint_bootstrap_n: int = 200
    dca_points: int = 80
    exact_curve_min_n: int = 10
    exact_curve_quantile: float = 0.975


PART1_SHEET_NAME_MAP = {
    "Summary": "Study_Summary",
    "Descriptive_Stats": "Table1_Descriptive_Stats",
    "Baseline": "Baseline_Summary_ByBestCut",
    "Table1_Baseline": "Baseline_Summary_ByBestCut",
    "ELN_Bins_Overall": "ELN_Bin_Survival_Overall",
    "ELN_Bins_Stage": "ELN_Bin_Survival_ByStage",
    "ELN_Bins_Surgery": "ELN_Bin_Survival_BySurg",
    "ELN_Exact_Curve": "ELN_Exact_OS_Curve",
    "ELN_Threshold": "ELN_LogRank_Thresholds",
    "ELN_Changepoint": "ELN_Changepoint_Profile",
    "ELN_Cox": "SEER_ELN_Cox_Model",
    "Table2_ELN_Cox": "SEER_ELN_Cox_Model",
    "ELN_PH": "ELN_PH_Test",
    "ELN_Cindex": "ELN_CIndex_Bootstrap",
    "ELN_Strategy_RMST": "Table2_SEER_RMST",
    "ELN_Strategy_RMST_AllMethods": "SEER_RMST_All_Methods",
    "ELN_Balance": "ELN_Weight_Balance",
    "ELN_Spline_Curve": "ELN_Spline_HR_Curve",
    "ELN_Calib_12m": "ELN_Calibration_12m",
    "ELN_Calib_36m": "ELN_Calibration_36m",
    "ELN_Calib_60m": "ELN_Calibration_60m",
    "ELN_DCA_12m": "ELN_DCA_12m",
    "ELN_DCA_36m": "ELN_DCA_36m",
    "ELN_DCA_60m": "ELN_DCA_60m",
    "POS_Cox": "SEER_POS_Cox_Model",
    "Table3_POS_Cox": "SEER_POS_Cox_Model",
    "POS_PH": "POS_PH_Test",
    "POS_Transition": "POS_Step_Effect",
    "POS_Curve": "POS_Spline_HR_Curve",
    "POS_RMST": "POS_Delta_RMST",
}

PART1_MAIN_FIGURES = [
    "Figure_1_SEER_ELN_Overview",
    "Figure_2_SEER_ELN_Modeling",
    "Figure_3_SEER_POS_Analysis",
]

PART1_SUPPLEMENTARY_FIGURES = [
    "Supplementary_Figure_S1_SEER_Model_Performance",
    "Supplementary_Figure_S2_SEER_ELN_By_Stage",
    "Supplementary_Figure_S3_SEER_RMST_OW",
]

PART1_MAIN_TABLE_SHEETS = [
    "Table1_Descriptive_Stats",
    "Table2_SEER_RMST",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    if not getattr(fig, "_skip_tight_layout", False):
        fig.tight_layout()
    stem = path.with_suffix("")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def choose_existing(categories: Sequence[str], series: pd.Series) -> List[str]:
    observed = set(series.astype(str).dropna().unique().tolist())
    ordered = [cat for cat in categories if cat in observed]
    extras = sorted(observed - set(ordered))
    return ordered + extras


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def sanitize_sheet_name(name: str) -> str:
    for token in ["[", "]", ":", "*", "?", "/", "\\"]:
        name = name.replace(token, "_")
    return name[:31]


def map_stage(value: object) -> str:
    s = str(value).strip().lower()
    if "localized" in s:
        return "Localized"
    if "regional" in s:
        return "Regional"
    if "distant" in s:
        return "Distant"
    return "Unknown"


def map_primary(value: object) -> str:
    s = str(value).lower()
    if "body" in s:
        return "Body"
    if "tail" in s:
        return "Tail"
    return "Other"


def map_sex(value: object) -> str:
    s = str(value).strip().lower()
    if "female" in s:
        return "Female"
    if "male" in s:
        return "Male"
    return "Unknown"


def map_chemo_bin(value: object) -> int:
    return 1 if "yes" in str(value).strip().lower() else 0


def map_radiation_bin(value: object) -> int:
    s = str(value).strip().lower()
    no_tokens = ("none", "unknown", "refused", "recommended")
    yes_tokens = ("beam", "radio", "isotope", "brachy")
    if any(token in s for token in yes_tokens) and not any(token in s for token in no_tokens):
        return 1
    return 0


def clean_surgery_extent(value: object) -> str:
    s = str(value).strip()
    if s.lower() == "other":
        return "Other"
    if s in {"Local", "Combined", "Extended"}:
        return s
    return "Other"


def km_summary_row(sub: pd.DataFrame, time_col: str, event_col: str, time_points: Iterable[int]) -> Dict[str, float]:
    row: Dict[str, float] = {"n": int(len(sub)), "Events": int(sub[event_col].sum()) if len(sub) else 0, "Median_OS_months": np.nan}
    for tau in time_points:
        row[f"S_{tau}m"] = np.nan
    if len(sub) == 0:
        return row
    kmf = KaplanMeierFitter()
    kmf.fit(sub[time_col], sub[event_col])
    row["Median_OS_months"] = float(kmf.median_survival_time_)
    for tau in time_points:
        try:
            surv = float(kmf.survival_function_at_times(tau).values[-1])
        except Exception:
            surv = float(kmf.predict(tau))
        row[f"S_{tau}m"] = surv
    return row


def rmst_from_kmf(kmf: KaplanMeierFitter, tau: float) -> float:
    sf = kmf.survival_function_
    col = sf.columns[0]
    t = sf.index.to_numpy(dtype=float)
    s = sf[col].to_numpy(dtype=float)
    if tau > t[-1]:
        t = np.append(t, tau)
        s = np.append(s, s[-1])
    else:
        mask = t <= tau
        t = t[mask]
        s = s[mask]
        if t[-1] < tau:
            t = np.append(t, tau)
            s = np.append(s, s[-1])
    return float(np.trapezoid(s, t))


def binary_event_by_tau(time: pd.Series, event: pd.Series, tau: int) -> pd.Series:
    t = to_numeric(time)
    e = to_numeric(event)
    return pd.Series(np.where((e == 1) & (t <= tau), 1, np.where(t > tau, 0, np.nan)), index=time.index)


def pretty_term(term: str) -> str:
    mapping = {
        "Age_per10": "Age (per 10 years)",
        "ELN_per5": "ELN (per 5 nodes)",
        "chemo_bin": "Chemotherapy: Yes vs No/Unknown",
        "radiation_bin": "Radiation: Any vs None/Unknown",
        "Stage_Regional": "Stage: Regional vs Localized",
        "Stage_Distant": "Stage: Distant vs Localized",
        "Stage_Unknown": "Stage: Unknown vs Localized",
        "Primary_part_Tail": "Primary site: Tail vs Body",
        "Primary_part_Other": "Primary site: Other vs Body",
        "Sex_clean_Male": "Sex: Male vs Female",
        "SurgeryExtent_clean_Combined": "Surgery: Combined vs Local",
        "SurgeryExtent_clean_Extended": "Surgery: Extended vs Local",
        "SurgeryExtent_clean_Other": "Surgery: Other vs Local",
        "POS_any": "POS >= 1 vs POS = 0",
        "C(Sex_clean)[T.Male]": "Sex: Male vs Female",
        "C(Sex_clean)[T.Unknown]": "Sex: Unknown vs Female",
        "C(Stage)[T.Regional]": "Stage: Regional vs Localized",
        "C(Stage)[T.Distant]": "Stage: Distant vs Localized",
        "C(Stage)[T.Unknown]": "Stage: Unknown vs Localized",
        "C(Primary_part)[T.Tail]": "Primary site: Tail vs Body",
        "C(Primary_part)[T.Other]": "Primary site: Other vs Body",
        "C(SurgeryExtent_clean)[T.Combined]": "Surgery: Combined vs Local",
        "C(SurgeryExtent_clean)[T.Extended]": "Surgery: Extended vs Local",
        "C(SurgeryExtent_clean)[T.Other]": "Surgery: Other vs Local",
    }
    if term in mapping:
        return mapping[term]
    if term.startswith("cr(POS_minus1"):
        return term.replace("cr(POS_minus1, 3)", "POS spline")
    if term.startswith("cr(ELN, 3)"):
        return term.replace("cr(ELN, 3)", "ELN spline")
    return term


def load_and_prepare_dataset(config: AnalysisConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    path = Path(config.excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    df = pd.read_excel(path, sheet_name=config.sheet_name)
    raw_n = len(df)

    df["SurvM"] = to_numeric(df["Survival months"])
    df["Event"] = df["Vital status recode (study cutoff used)"].astype(str).str.contains("dead", case=False, na=False).astype(int)
    df["ELN"] = to_numeric(df["Regional nodes examined (1988+)"])
    df["POS"] = to_numeric(df["Regional nodes positive (1988+)"])
    df["Age"] = to_numeric(df["Age"])
    df["Year"] = to_numeric(df["Year of diagnosis"])
    df["Stage"] = df["Summary stage 2000 (1998-2017)"].map(map_stage)
    df["Primary_part"] = df["Primary Site - labeled"].map(map_primary)
    df["Sex_clean"] = df["Sex"].map(map_sex)
    df["Sex_bin"] = np.where(df["Sex_clean"] == "Male", 1, 0)
    df["chemo_bin"] = df["Chemotherapy recode (yes, no/unk)"].map(map_chemo_bin)
    df["radiation_bin"] = df["Radiation recode"].map(map_radiation_bin)
    df["SurgeryExtent_clean"] = df["SurgeryExtent"].map(clean_surgery_extent)

    df.loc[df["POS"].isin([90, 95, 97, 98, 99, 126]), "POS"] = np.nan
    df.loc[df["ELN"] < 0, "ELN"] = np.nan
    df.loc[df["POS"] < 0, "POS"] = np.nan

    bad_relation = df["POS"].notna() & df["ELN"].notna() & (df["POS"] > df["ELN"])
    removed_pos_gt_eln = int(bad_relation.sum())
    if removed_pos_gt_eln:
        df = df.loc[~bad_relation].copy()

    df["LNR"] = np.where((df["ELN"] > 0) & (df["POS"] >= 0), df["POS"] / df["ELN"], np.nan)
    df["POS_minus1"] = (df["POS"] - 1).clip(lower=0)
    df["POS_any"] = (df["POS"] > 0).astype(int)
    df["Age_per10"] = df["Age"] / 10.0
    df["ELN_per5"] = df["ELN"] / 5.0

    usable = (
        df["SurvM"].notna()
        & (df["SurvM"] > 0)
        & df["ELN"].notna()
        & (df["ELN"] > 0)
        & df["POS"].notna()
        & (df["POS"] >= 0)
    )
    analysis = df.loc[usable].copy()

    for col, order in [
        ("Stage", ["Localized", "Regional", "Distant", "Unknown"]),
        ("Primary_part", ["Body", "Tail", "Other"]),
        ("Sex_clean", ["Female", "Male", "Unknown"]),
        ("SurgeryExtent_clean", ["Local", "Combined", "Extended", "Other"]),
    ]:
        analysis[col] = pd.Categorical(analysis[col], categories=choose_existing(order, analysis[col]))

    summary = {
        "raw_n": raw_n,
        "removed_pos_gt_eln": removed_pos_gt_eln,
        "analysis_n": int(len(analysis)),
        "event_n": int(analysis["Event"].sum()),
        "censored_n": int((analysis["Event"] == 0).sum()),
        "stage_unknown_n": int((analysis["Stage"].astype(str) == "Unknown").sum()),
    }
    return analysis, summary


def build_baseline_table(df: pd.DataFrame, cut: int) -> pd.DataFrame:
    d = df.copy()
    d["ELN_group"] = np.where(d["ELN"] >= cut, f"ELN>={cut}", f"ELN<{cut}")
    rows: List[Dict[str, object]] = []

    def add_row(variable: str, overall: str, high: str, low: str) -> None:
        rows.append({"Variable": variable, "Overall": overall, f"ELN>={cut}": high, f"ELN<{cut}": low})

    for variable in ["Age", "ELN", "POS", "LNR", "SurvM"]:
        s = to_numeric(d[variable])
        high = s[d["ELN_group"] == f"ELN>={cut}"]
        low = s[d["ELN_group"] == f"ELN<{cut}"]
        overall_text = f"{np.nanmedian(s):.2f} ({np.nanquantile(s, 0.25):.2f}-{np.nanquantile(s, 0.75):.2f})"
        high_text = f"{np.nanmedian(high):.2f} ({np.nanquantile(high, 0.25):.2f}-{np.nanquantile(high, 0.75):.2f})"
        low_text = f"{np.nanmedian(low):.2f} ({np.nanquantile(low, 0.25):.2f}-{np.nanquantile(low, 0.75):.2f})"
        add_row(variable, overall_text, high_text, low_text)

    for variable in ["Sex_clean", "Stage", "Primary_part", "SurgeryExtent_clean"]:
        overall = "; ".join([f"{k} {v} ({v / len(d):.1%})" for k, v in d[variable].astype(str).value_counts().items()])
        high_sub = d.loc[d["ELN_group"] == f"ELN>={cut}", variable].astype(str)
        low_sub = d.loc[d["ELN_group"] == f"ELN<{cut}", variable].astype(str)
        high = "; ".join([f"{k} {v} ({v / len(high_sub):.1%})" for k, v in high_sub.value_counts().items()])
        low = "; ".join([f"{k} {v} ({v / len(low_sub):.1%})" for k, v in low_sub.value_counts().items()])
        add_row(variable, overall, high, low)

    for variable in ["chemo_bin", "radiation_bin"]:
        label_map = {1: "Yes", 0: "No"}
        overall = "; ".join([f"{label_map.get(k, k)} {v} ({v / len(d):.1%})" for k, v in d[variable].value_counts().items()])
        high_sub = d.loc[d["ELN_group"] == f"ELN>={cut}", variable]
        low_sub = d.loc[d["ELN_group"] == f"ELN<{cut}", variable]
        high = "; ".join([f"{label_map.get(k, k)} {v} ({v / len(high_sub):.1%})" for k, v in high_sub.value_counts().items()])
        low = "; ".join([f"{label_map.get(k, k)} {v} ({v / len(low_sub):.1%})" for k, v in low_sub.value_counts().items()])
        add_row(variable, overall, high, low)
    return pd.DataFrame(rows)


def compute_binary_smd(p_low: float, p_high: float) -> float:
    denom = np.sqrt((p_low * (1 - p_low) + p_high * (1 - p_high)) / 2)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(abs(p_high - p_low) / denom)


def compute_multilevel_smd(low_counts: pd.Series, high_counts: pd.Series, levels: Sequence[str]) -> float:
    low_total = float(low_counts.sum())
    high_total = float(high_counts.sum())
    if low_total == 0 or high_total == 0:
        return np.nan
    smds = []
    for level in levels:
        p_low = float(low_counts.get(level, 0)) / low_total
        p_high = float(high_counts.get(level, 0)) / high_total
        smds.append(compute_binary_smd(p_low, p_high))
    return float(np.sqrt(np.nanmean(np.square(smds)))) if smds else np.nan


def format_count_pct(value: int, total: int) -> str:
    if total == 0:
        return "0 (0.0%)"
    return f"{int(value)} ({value / total * 100:.1f}%)"


def build_manuscript_descriptive_table(df: pd.DataFrame, cut: int) -> pd.DataFrame:
    d = df.copy()
    d["ELN_group"] = np.where(d["ELN"] >= cut, f"ELN≥{cut}", f"ELN<{cut}")
    low_label = f"ELN<{cut}"
    high_label = f"ELN≥{cut}"
    low = d.loc[d["ELN_group"] == low_label].copy()
    high = d.loc[d["ELN_group"] == high_label].copy()

    if "Race recode (W, B, AI, API)" in d.columns:
        d["Race_group"] = d["Race recode (W, B, AI, API)"].astype(str).replace(
            {
                "Asian or Pacific Islander": "Asian/Pacific Islander",
                "American Indian/Alaska Native": "AI/AN",
                "Blank(s)": "Unknown",
            }
        )
        low["Race_group"] = low["Race recode (W, B, AI, API)"].astype(str).replace(
            {
                "Asian or Pacific Islander": "Asian/Pacific Islander",
                "American Indian/Alaska Native": "AI/AN",
                "Blank(s)": "Unknown",
            }
        )
        high["Race_group"] = high["Race recode (W, B, AI, API)"].astype(str).replace(
            {
                "Asian or Pacific Islander": "Asian/Pacific Islander",
                "American Indian/Alaska Native": "AI/AN",
                "Blank(s)": "Unknown",
            }
        )
    else:
        d["Race_group"] = "Unknown"
        low["Race_group"] = "Unknown"
        high["Race_group"] = "Unknown"

    d["Age_group"] = pd.qcut(d["Age"], q=4, duplicates="drop")
    low["Age_group"] = pd.Categorical(pd.cut(low["Age"], bins=np.unique(np.quantile(d["Age"], [0, 0.25, 0.5, 0.75, 1])) , include_lowest=True))
    high["Age_group"] = pd.Categorical(pd.cut(high["Age"], bins=np.unique(np.quantile(d["Age"], [0, 0.25, 0.5, 0.75, 1])) , include_lowest=True))

    rows: List[Dict[str, object]] = [
        {"变量/分类": "", low_label: "", high_label: "", "P value": f"(<{cut} vs ≥{cut})", "SMD": f"(<{cut} vs ≥{cut})"}
    ]

    def add_categorical_block(title: str, variable: str, levels: Sequence[object], p_value: float, smd: float) -> None:
        rows.append(
            {
                "变量/分类": title,
                low_label: "",
                high_label: "",
                "P value": "" if np.isnan(p_value) else f"{p_value:.4f}",
                "SMD": "" if np.isnan(smd) else f"{smd:.3f}",
            }
        )
        low_counts = low[variable].astype(str).value_counts()
        high_counts = high[variable].astype(str).value_counts()
        for level in levels:
            level_str = str(level)
            rows.append(
                {
                    "变量/分类": level_str,
                    low_label: format_count_pct(int(low_counts.get(level_str, 0)), len(low)),
                    high_label: format_count_pct(int(high_counts.get(level_str, 0)), len(high)),
                    "P value": "",
                    "SMD": "",
                }
            )

    age_bins = np.unique(np.quantile(d["Age"], [0, 0.25, 0.5, 0.75, 1]))
    d["Age_group"] = pd.cut(d["Age"], bins=age_bins, include_lowest=True)
    low["Age_group"] = pd.cut(low["Age"], bins=age_bins, include_lowest=True)
    high["Age_group"] = pd.cut(high["Age"], bins=age_bins, include_lowest=True)
    age_p = mannwhitneyu(low["Age"], high["Age"], alternative="two-sided").pvalue if len(low) and len(high) else np.nan
    age_smd = compute_binary_smd(float(low["Age"].mean() < high["Age"].mean()), float(high["Age"].mean() > low["Age"].mean()))
    # Overwrite the binary placeholder with a proper continuous SMD.
    pooled_sd = np.sqrt((np.nanvar(low["Age"], ddof=1) + np.nanvar(high["Age"], ddof=1)) / 2)
    if pooled_sd and not np.isnan(pooled_sd):
        age_smd = float(abs(np.nanmean(high["Age"]) - np.nanmean(low["Age"])) / pooled_sd)
    add_categorical_block("Age", "Age_group", [str(x) for x in d["Age_group"].cat.categories], age_p, age_smd)

    categorical_specs = [
        ("Sex", "Sex_clean", ["Female", "Male", "Unknown"]),
        ("Race", "Race_group", ["White", "Black", "Asian/Pacific Islander", "AI/AN", "Unknown"]),
        ("Stage", "Stage", ["Localized", "Regional", "Distant", "Unknown"]),
        ("Primary site", "Primary_part", ["Body", "Tail", "Other"]),
        ("Surgery extent", "SurgeryExtent_clean", ["Local", "Combined", "Extended", "Other"]),
    ]
    for title, variable, levels in categorical_specs:
        contingency = pd.crosstab(d[variable].astype(str), d["ELN_group"])
        p_value = chi2_contingency(contingency)[1] if contingency.shape[0] > 1 and contingency.shape[1] > 1 else np.nan
        low_counts = low[variable].astype(str).value_counts()
        high_counts = high[variable].astype(str).value_counts()
        smd = compute_multilevel_smd(low_counts, high_counts, [str(x) for x in levels])
        add_categorical_block(title, variable, levels, p_value, smd)

    for title, variable in [("Radiation", "radiation_bin"), ("Chemotherapy", "chemo_bin")]:
        d[title] = d[variable].map({1: "Yes", 0: "No/Unknown"}).fillna("No/Unknown")
        low[title] = low[variable].map({1: "Yes", 0: "No/Unknown"}).fillna("No/Unknown")
        high[title] = high[variable].map({1: "Yes", 0: "No/Unknown"}).fillna("No/Unknown")
        contingency = pd.crosstab(d[title], d["ELN_group"])
        p_value = chi2_contingency(contingency)[1] if contingency.shape[0] > 1 and contingency.shape[1] > 1 else np.nan
        low_counts = low[title].value_counts()
        high_counts = high[title].value_counts()
        smd = compute_multilevel_smd(low_counts, high_counts, ["Yes", "No/Unknown"])
        add_categorical_block(title, title, ["Yes", "No/Unknown"], p_value, smd)

    return pd.DataFrame(rows)


def summarize_eln_bins(df: pd.DataFrame, config: AnalysisConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["ELN_bin_detailed"] = pd.cut(d["ELN"], bins=config.eln_bins_detailed, labels=config.eln_bin_labels_detailed)
    d["ELN_bin_plot"] = pd.cut(d["ELN"], bins=config.eln_bins_plot, labels=config.eln_bin_labels_plot)

    overall_rows = []
    for label in config.eln_bin_labels_detailed:
        sub = d.loc[d["ELN_bin_detailed"] == label]
        row = km_summary_row(sub, "SurvM", "Event", config.calibration_times)
        row["ELN_bin"] = label
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)[["ELN_bin", "n", "Events", "Median_OS_months", "S_12m", "S_36m", "S_60m"]]

    stage_rows = []
    for stage in ["Localized", "Regional", "Distant", "Unknown"]:
        sub_stage = d.loc[d["Stage"].astype(str) == stage]
        for label in config.eln_bin_labels_detailed:
            sub = sub_stage.loc[sub_stage["ELN_bin_detailed"] == label]
            row = km_summary_row(sub, "SurvM", "Event", config.calibration_times)
            row["Stage"] = stage
            row["ELN_bin"] = label
            stage_rows.append(row)
    by_stage = pd.DataFrame(stage_rows)[["Stage", "ELN_bin", "n", "Events", "Median_OS_months", "S_12m", "S_36m", "S_60m"]]

    surgery_rows = []
    for surgery in ["Local", "Combined", "Extended", "Other"]:
        sub_surgery = d.loc[d["SurgeryExtent_clean"].astype(str) == surgery]
        for label in config.eln_bin_labels_detailed:
            sub = sub_surgery.loc[sub_surgery["ELN_bin_detailed"] == label]
            row = km_summary_row(sub, "SurvM", "Event", config.calibration_times)
            row["SurgeryExtent"] = surgery
            row["ELN_bin"] = label
            surgery_rows.append(row)
    by_surgery = pd.DataFrame(surgery_rows)[["SurgeryExtent", "ELN_bin", "n", "Events", "Median_OS_months", "S_12m", "S_36m", "S_60m"]]

    exact_rows = []
    for eln_value in sorted(d["ELN"].dropna().astype(int).unique()):
        sub = d.loc[d["ELN"] == eln_value]
        row = km_summary_row(sub, "SurvM", "Event", [])
        exact_rows.append({"ELN": int(eln_value), "n": int(len(sub)), "Median_OS_months": row["Median_OS_months"]})
    exact_curve = pd.DataFrame(exact_rows).sort_values("ELN")
    if not exact_curve.empty:
        weighted_num = (exact_curve["Median_OS_months"] * exact_curve["n"]).rolling(5, center=True, min_periods=1).sum()
        weighted_den = exact_curve["n"].rolling(5, center=True, min_periods=1).sum()
        exact_curve["Median_OS_smoothed"] = weighted_num / weighted_den
    return overall, by_stage, by_surgery, exact_curve


def threshold_search(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows = []
    for cut in config.threshold_candidates:
        high = df["ELN"] >= cut
        low = df["ELN"] < cut
        if int(high.sum()) < 30 or int(low.sum()) < 30:
            continue
        res = multivariate_logrank_test(df["SurvM"], np.where(high, f">={cut}", f"<{cut}"), df["Event"])
        rows.append({"cut": int(cut), "logrank_stat": float(res.test_statistic), "p_value": float(res.p_value), "n_high": int(high.sum()), "n_low": int(low.sum())})
    return pd.DataFrame(rows).sort_values("cut").reset_index(drop=True)


def changepoint_search(df: pd.DataFrame, config: AnalysisConfig) -> Tuple[pd.DataFrame, int, Tuple[float, float]]:
    results = []
    for cut in config.changepoint_candidates:
        tmp = df[["SurvM", "Event", "ELN"]].copy()
        tmp["ELN_hinge"] = np.maximum(0.0, tmp["ELN"] - float(cut))
        if int((tmp["ELN_hinge"] > 0).sum()) < 5:
            continue
        cph = CoxPHFitter(penalizer=0.05)
        try:
            cph.fit(tmp, duration_col="SurvM", event_col="Event", robust=True)
            results.append({"cut": int(cut), "loglik": float(cph.log_likelihood_)})
        except Exception:
            continue
    if not results:
        return pd.DataFrame(columns=["cut", "loglik"]), np.nan, (np.nan, np.nan)

    grid = pd.DataFrame(results).sort_values("cut").reset_index(drop=True)
    best_cut = int(grid.loc[grid["loglik"].idxmax(), "cut"])

    rng = np.random.default_rng(config.random_state)
    idx = np.arange(len(df))
    boot_cuts = []
    for _ in range(config.changepoint_bootstrap_n):
        sample = df.iloc[rng.choice(idx, size=len(idx), replace=True)][["SurvM", "Event", "ELN"]].copy()
        best_ll = -np.inf
        best_local = None
        for cut in config.changepoint_candidates:
            sample["ELN_hinge"] = np.maximum(0.0, sample["ELN"] - float(cut))
            if int((sample["ELN_hinge"] > 0).sum()) < 5:
                continue
            cph = CoxPHFitter(penalizer=0.05)
            try:
                cph.fit(sample, duration_col="SurvM", event_col="Event", robust=True)
                if cph.log_likelihood_ > best_ll:
                    best_ll = cph.log_likelihood_
                    best_local = int(cut)
            except Exception:
                continue
        if best_local is not None:
            boot_cuts.append(best_local)
    ci = tuple(np.percentile(boot_cuts, [2.5, 97.5]).tolist()) if boot_cuts else (np.nan, np.nan)
    return grid, best_cut, ci


def build_main_cox_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    for col, order in [
        ("Stage", ["Localized", "Regional", "Distant", "Unknown"]),
        ("Primary_part", ["Body", "Tail", "Other"]),
        ("Sex_clean", ["Female", "Male", "Unknown"]),
        ("SurgeryExtent_clean", ["Local", "Combined", "Extended", "Other"]),
    ]:
        d[col] = pd.Categorical(d[col], categories=choose_existing(order, d[col]))

    covars = ["Age_per10", "ELN_per5", "chemo_bin", "radiation_bin", "Stage", "Primary_part", "Sex_clean", "SurgeryExtent_clean"]
    X = pd.get_dummies(d[covars], drop_first=True)
    fit = pd.concat([d[["SurvM", "Event", "ELN"]], X], axis=1).dropna().rename(columns={"SurvM": "T", "Event": "E"})
    fit["_ELN_source"] = fit["ELN"]
    fit = fit.drop(columns=["ELN"])
    model_cols = [c for c in fit.columns if c not in {"T", "E", "_ELN_source"}]
    return fit, model_cols


def fit_main_cox(df: pd.DataFrame, config: AnalysisConfig) -> Tuple[CoxPHFitter, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit, model_cols = build_main_cox_dataset(df)
    cph = None
    for pen in [0.05, 0.1, 0.3, 0.5, 1.0]:
        try:
            cph_try = CoxPHFitter(penalizer=pen)
            cph_try.fit(fit[["T", "E"] + model_cols], duration_col="T", event_col="E", robust=True)
            cph = cph_try
            break
        except ConvergenceError:
            continue
    if cph is None:
        raise RuntimeError("Main Cox model failed to converge.")

    summary = cph.summary.reset_index().rename(columns={"index": "covariate"})
    summary["pretty_term"] = summary["covariate"].map(pretty_term)

    ph = proportional_hazard_test(cph, fit[["T", "E"] + model_cols], time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
    ph["pretty_term"] = ph["covariate"].map(pretty_term)

    rng = np.random.default_rng(config.random_state)
    values = []
    base_data = fit[["T", "E"] + model_cols].copy()
    for _ in range(config.cindex_bootstrap_n):
        sample = base_data.loc[rng.choice(base_data.index, size=len(base_data), replace=True)]
        try:
            cph_b = CoxPHFitter(penalizer=cph.penalizer)
            cph_b.fit(sample, duration_col="T", event_col="E", robust=True)
            pred = cph_b.predict_partial_hazard(sample)
            values.append(concordance_index(sample["T"], -pred.values, sample["E"]))
        except Exception:
            continue
    cindex_df = pd.DataFrame({"Bootstrap_Cindex": values})
    fit.attrs["model_cols"] = model_cols
    return cph, fit, summary, ph, cindex_df


def build_eln_spline_model(df: pd.DataFrame) -> Tuple[CoxPHFitter, pd.DataFrame, object]:
    d = df.copy()
    for col, order in [
        ("Stage", ["Localized", "Regional", "Distant", "Unknown"]),
        ("Primary_part", ["Body", "Tail", "Other"]),
        ("Sex_clean", ["Female", "Male", "Unknown"]),
        ("SurgeryExtent_clean", ["Local", "Combined", "Extended", "Other"]),
    ]:
        d[col] = pd.Categorical(d[col], categories=choose_existing(order, d[col]))

    formula = (
        "bs(ELN, df=4, degree=3, include_intercept=False)"
        " + Age_per10 + chemo_bin + radiation_bin"
        " + C(Sex_clean) + C(Stage) + C(Primary_part) + C(SurgeryExtent_clean)"
    )
    design_matrix = dmatrix(formula, d)
    design_info = design_matrix.design_info
    design = pd.DataFrame(np.asarray(design_matrix), columns=design_info.column_names, index=d.index)
    if "Intercept" in design.columns:
        design = design.drop(columns=["Intercept"])
    fit = pd.concat([d[["SurvM", "Event"]], design], axis=1).dropna().rename(columns={"SurvM": "T", "Event": "E"})

    cph = None
    for pen in [0.05, 0.1, 0.3, 0.5]:
        try:
            cph_try = CoxPHFitter(penalizer=pen)
            cph_try.fit(fit, duration_col="T", event_col="E", robust=True)
            cph = cph_try
            break
        except Exception:
            continue
    if cph is None:
        raise RuntimeError("Adjusted ELN spline model failed to converge.")
    return cph, fit, design_info


def predict_relative_hr_curve(cph: CoxPHFitter, design_info, base_profile: pd.DataFrame, varying_col: str, grid: np.ndarray, reference_value: float) -> pd.DataFrame:
    profile_grid = pd.concat([base_profile] * len(grid), ignore_index=True)
    profile_grid[varying_col] = grid
    design_grid = build_design_matrices([design_info], profile_grid, return_type="dataframe")[0]
    if "Intercept" in design_grid.columns:
        design_grid = design_grid.drop(columns=["Intercept"])

    profile_ref = base_profile.copy()
    profile_ref[varying_col] = reference_value
    design_ref = build_design_matrices([design_info], profile_ref, return_type="dataframe")[0]
    if "Intercept" in design_ref.columns:
        design_ref = design_ref.drop(columns=["Intercept"])

    beta = cph.params_.loc[design_grid.columns].to_numpy()
    cov = cph.variance_matrix_.loc[design_grid.columns, design_grid.columns].to_numpy()
    ref_vec = design_ref.iloc[0].to_numpy()

    rows = []
    for x_value, row in zip(grid, design_grid.to_numpy()):
        delta = row - ref_vec
        lp = float(delta @ beta)
        se = float(np.sqrt(max(delta @ cov @ delta.T, 0.0)))
        rows.append({"ELN" if varying_col == "ELN" else varying_col: float(x_value), "HR": math.exp(lp), "HR_lower": math.exp(lp - 1.96 * se), "HR_upper": math.exp(lp + 1.96 * se)})
    return pd.DataFrame(rows)


def risk_from_cox(cph: CoxPHFitter, X: pd.DataFrame, tau: int) -> np.ndarray:
    tt = cph.baseline_survival_.index.to_numpy(dtype=float)
    t_star = tt[np.argmin(np.abs(tt - tau))]
    s0 = float(cph.baseline_survival_.loc[t_star].values.item())
    lp = cph.predict_partial_hazard(X).values.flatten()
    return 1.0 - np.power(s0, lp)


def generate_calibration_and_dca(cph: CoxPHFitter, fit_data: pd.DataFrame, config: AnalysisConfig) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    model_cols = fit_data.attrs["model_cols"]
    X = fit_data.loc[:, model_cols]
    calibration_tables: Dict[int, pd.DataFrame] = {}
    dca_tables: Dict[int, pd.DataFrame] = {}

    for tau in config.calibration_times:
        y = binary_event_by_tau(fit_data["T"], fit_data["E"], tau)
        mask = ~y.isna()
        y_bin = y.loc[mask].astype(int)
        X_bin = X.loc[mask]
        risk = risk_from_cox(cph, X_bin, tau)

        calib = pd.DataFrame({"risk": risk, "event": y_bin.values})
        calib["decile"] = pd.qcut(calib["risk"], 10, duplicates="drop")
        calib_tab = calib.groupby("decile", observed=False).agg(predicted=("risk", "mean"), observed=("event", "mean"), n=("event", "size")).reset_index()
        calibration_tables[tau] = calib_tab

        thresholds = np.linspace(0.01, 0.80, config.dca_points)
        rows = []
        y_arr = y_bin.to_numpy()
        n_total = len(y_arr)
        for pt in thresholds:
            pred_pos = (risk >= pt).astype(int)
            tp = np.sum((pred_pos == 1) & (y_arr == 1))
            fp = np.sum((pred_pos == 1) & (y_arr == 0))
            nb_model = (tp / n_total) - (fp / n_total) * (pt / (1 - pt))
            tp_all = np.sum(y_arr == 1)
            fp_all = np.sum(y_arr == 0)
            nb_all = (tp_all / n_total) - (fp_all / n_total) * (pt / (1 - pt))
            rows.append({"threshold": pt, "net_benefit_model": nb_model, "net_benefit_all": nb_all, "net_benefit_none": 0.0})
        dca_tables[tau] = pd.DataFrame(rows)
    return calibration_tables, dca_tables


def build_ps_matrix(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col, order in [
        ("Stage", ["Localized", "Regional", "Distant", "Unknown"]),
        ("Primary_part", ["Body", "Tail", "Other"]),
        ("Sex_clean", ["Female", "Male", "Unknown"]),
        ("SurgeryExtent_clean", ["Local", "Combined", "Extended", "Other"]),
    ]:
        d[col] = pd.Categorical(d[col], categories=choose_existing(order, d[col]))
    X = pd.get_dummies(d[["Age_per10", "Stage", "Primary_part", "Sex_clean", "SurgeryExtent_clean", "chemo_bin", "radiation_bin"]], drop_first=True)
    return X.fillna(0)


def _weighted_mean_var(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    if len(x) == 0 or len(w) == 0 or float(np.sum(w)) <= 0:
        return np.nan, np.nan
    mu = float(np.average(x, weights=w))
    var = float(np.average((x - mu) ** 2, weights=w))
    return mu, var


def _smd_numeric(x_t: np.ndarray, x_c: np.ndarray, w_t: Optional[np.ndarray], w_c: Optional[np.ndarray]) -> float:
    if w_t is None or w_c is None:
        mt, mc = np.nanmean(x_t), np.nanmean(x_c)
        vt, vc = np.nanvar(x_t, ddof=1), np.nanvar(x_c, ddof=1)
    else:
        mt, vt = _weighted_mean_var(x_t, w_t)
        mc, vc = _weighted_mean_var(x_c, w_c)
    pooled = np.sqrt((vt + vc) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((mt - mc) / pooled)


def compute_balance_table(df_ps: pd.DataFrame, covars: List[str]) -> pd.DataFrame:
    rows = []
    for variable in covars:
        if variable not in df_ps.columns:
            continue
        if pd.api.types.is_numeric_dtype(df_ps[variable]):
            matrices = {variable: df_ps[variable].to_numpy()}
        else:
            dummies = pd.get_dummies(df_ps[variable].astype(str), prefix=variable)
            matrices = {col: dummies[col].to_numpy() for col in dummies.columns}
        for label, values in matrices.items():
            x_t = values[df_ps["TREAT"] == 1]
            x_c = values[df_ps["TREAT"] == 0]
            w_t = df_ps.loc[df_ps["TREAT"] == 1, "W"].to_numpy()
            w_c = df_ps.loc[df_ps["TREAT"] == 0, "W"].to_numpy()
            rows.append({"Variable": label, "SMD_before": _smd_numeric(x_t, x_c, None, None), "SMD_after": _smd_numeric(x_t, x_c, w_t, w_c)})
    return pd.DataFrame(rows)


def fit_strategy_weights(df: pd.DataFrame, cut: int, method: str) -> pd.DataFrame:
    d = df.copy()
    d["TREAT"] = (d["ELN"] >= cut).astype(int)
    X = build_ps_matrix(d)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, d["TREAT"])
    ps = np.clip(model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    d["PS"] = ps
    if method.upper() == "IPTW":
        p_t = float(d["TREAT"].mean())
        d["W"] = np.where(d["TREAT"] == 1, p_t / ps, (1 - p_t) / (1 - ps))
    else:
        d["W"] = np.where(d["TREAT"] == 1, 1 - ps, ps)
    return d


def run_strategy_analysis(df: pd.DataFrame, config: AnalysisConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rmst_rows = []
    balance_rows = []
    for cut in config.strategy_cuts:
        for method in config.strategy_methods:
            weighted = fit_strategy_weights(df, cut, method)
            balance = compute_balance_table(weighted, ["Age_per10", "Stage", "Primary_part", "Sex_clean", "SurgeryExtent_clean", "chemo_bin", "radiation_bin"])
            balance["cut"] = cut
            balance["method"] = method
            balance_rows.append(balance)

            for tau in config.calibration_times:
                km_low = KaplanMeierFitter().fit(weighted.loc[weighted["TREAT"] == 0, "SurvM"], weighted.loc[weighted["TREAT"] == 0, "Event"], weights=weighted.loc[weighted["TREAT"] == 0, "W"])
                km_high = KaplanMeierFitter().fit(weighted.loc[weighted["TREAT"] == 1, "SurvM"], weighted.loc[weighted["TREAT"] == 1, "Event"], weights=weighted.loc[weighted["TREAT"] == 1, "W"])
                rmst_rows.append({"cut": cut, "method": method, "tau_m": tau, "RMST_high": rmst_from_kmf(km_high, tau), "RMST_low": rmst_from_kmf(km_low, tau), "Delta_RMST": rmst_from_kmf(km_high, tau) - rmst_from_kmf(km_low, tau)})
    return pd.DataFrame(rmst_rows), pd.concat(balance_rows, ignore_index=True)


def select_preferred_weight_method(balance_df: pd.DataFrame) -> str:
    if balance_df.empty:
        return "OW"
    ranked = (
        balance_df.assign(abs_after=balance_df["SMD_after"].abs())
        .groupby("method", as_index=False)["abs_after"]
        .mean()
        .sort_values(["abs_after", "method"])
    )
    return str(ranked.iloc[0]["method"])


def build_pos_model(df: pd.DataFrame) -> Tuple[CoxPHFitter, pd.DataFrame, object, str]:
    d = df.copy()
    for col, order in [
        ("Stage", ["Localized", "Regional", "Distant", "Unknown"]),
        ("Primary_part", ["Body", "Tail", "Other"]),
        ("Sex_clean", ["Female", "Male", "Unknown"]),
        ("SurgeryExtent_clean", ["Local", "Combined", "Extended", "Other"]),
    ]:
        d[col] = pd.Categorical(d[col], categories=choose_existing(order, d[col]))

    formula = (
        "cr(POS_minus1, 3) + cr(ELN, 3) + POS_any"
        " + Age_per10 + chemo_bin + radiation_bin"
        " + C(Sex_clean) + C(Stage) + C(Primary_part) + C(SurgeryExtent_clean)"
    )
    design_matrix = dmatrix(formula, d)
    design_info = design_matrix.design_info
    design = pd.DataFrame(np.asarray(design_matrix), columns=design_info.column_names, index=d.index)
    if "Intercept" in design.columns:
        design = design.drop(columns=["Intercept"])
    design = design.loc[:, design.std(axis=0) > 0]
    fit = pd.concat([d[["SurvM", "Event"]], design], axis=1).dropna().rename(columns={"SurvM": "T", "Event": "E"})

    cph = None
    for pen in [0.1, 0.3, 0.5, 1.0]:
        try:
            cph_try = CoxPHFitter(penalizer=pen)
            cph_try.fit(fit, duration_col="T", event_col="E", robust=True)
            cph = cph_try
            break
        except Exception:
            continue
    if cph is None:
        raise RuntimeError("POS model failed to converge.")
    return cph, fit, design_info, formula


def build_pos_reference_profile(df: pd.DataFrame) -> pd.DataFrame:
    ref = {
        "POS_minus1": 0,
        "POS_any": 0,
        "ELN": float(df["ELN"].median()),
        "Age_per10": float(df["Age_per10"].median()),
        "chemo_bin": int(df["chemo_bin"].mode().iloc[0]),
        "radiation_bin": int(df["radiation_bin"].mode().iloc[0]),
        "Sex_clean": str(df["Sex_clean"].mode().iloc[0]),
        "Stage": str(df["Stage"].mode().iloc[0]),
        "Primary_part": str(df["Primary_part"].mode().iloc[0]),
        "SurgeryExtent_clean": str(df["SurgeryExtent_clean"].mode().iloc[0]),
    }
    return pd.DataFrame([ref])


def predict_pos_effect_curve(cph: CoxPHFitter, design_info, reference_profile: pd.DataFrame, pos_grid: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = reference_profile.copy()
    base["POS_minus1"] = 0
    base["POS_any"] = 0
    design_ref = build_design_matrices([design_info], base, return_type="dataframe")[0]
    if "Intercept" in design_ref.columns:
        design_ref = design_ref.drop(columns=["Intercept"])

    beta = cph.params_.loc[design_ref.columns].to_numpy()
    cov = cph.variance_matrix_.loc[design_ref.columns, design_ref.columns].to_numpy()
    ref_vec = design_ref.iloc[0].to_numpy()

    step = reference_profile.copy()
    step["POS_minus1"] = 0
    step["POS_any"] = 1
    design_step = build_design_matrices([design_info], step, return_type="dataframe")[0]
    if "Intercept" in design_step.columns:
        design_step = design_step.drop(columns=["Intercept"])
    delta_step = design_step.iloc[0].to_numpy() - ref_vec
    lp_step = float(delta_step @ beta)
    se_step = float(np.sqrt(max(delta_step @ cov @ delta_step.T, 0.0)))
    transition = pd.DataFrame(
        [
            {"group": "POS=0", "HR": 1.0, "HR_lower": 1.0, "HR_upper": 1.0},
            {"group": "POS>=1", "HR": math.exp(lp_step), "HR_lower": math.exp(lp_step - 1.96 * se_step), "HR_upper": math.exp(lp_step + 1.96 * se_step)},
        ]
    )

    curve_rows = []
    for pos in pos_grid:
        row = reference_profile.copy()
        row["POS_any"] = int(pos > 0)
        row["POS_minus1"] = max(pos - 1, 0)
        design_pos = build_design_matrices([design_info], row, return_type="dataframe")[0]
        if "Intercept" in design_pos.columns:
            design_pos = design_pos.drop(columns=["Intercept"])
        delta = design_pos.iloc[0].to_numpy() - ref_vec
        lp = float(delta @ beta)
        se = float(np.sqrt(max(delta @ cov @ delta.T, 0.0)))
        curve_rows.append({"POS": int(pos), "HR": math.exp(lp), "HR_lower": math.exp(lp - 1.96 * se), "HR_upper": math.exp(lp + 1.96 * se)})
    return transition, pd.DataFrame(curve_rows)


def pos_tier(value: float) -> str:
    if value <= 0:
        return "POS=0"
    if value <= 2:
        return "POS=1-2"
    if value <= 4:
        return "POS=3-4"
    return "POS>=5"


def compute_pos_weighted_rmst(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["POS_tier"] = d["POS"].map(pos_tier)
    formula = (
        "bs(ELN, df=4, degree=3, include_intercept=False)"
        " + Age_per10 + chemo_bin + radiation_bin"
        " + C(Sex_clean) + C(Stage) + C(Primary_part) + C(SurgeryExtent_clean)"
    )
    rows = []
    baseline = d.loc[d["POS_tier"] == "POS=0"].copy()
    for tier in ["POS=1-2", "POS=3-4", "POS>=5"]:
        contrast = pd.concat([baseline, d.loc[d["POS_tier"] == tier]], axis=0).copy()
        contrast["tier_binary"] = (contrast["POS_tier"] == tier).astype(int)
        X = dmatrix(formula, contrast, return_type="dataframe")
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])
        model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
        model.fit(X, contrast["tier_binary"])
        ps = np.clip(model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
        weights = np.where(contrast["tier_binary"] == 1, 1 - ps, ps)

        for tau in (12, 36, 60):
            km_control = KaplanMeierFitter().fit(contrast.loc[contrast["tier_binary"] == 0, "SurvM"], contrast.loc[contrast["tier_binary"] == 0, "Event"], weights=weights[contrast["tier_binary"] == 0])
            km_treat = KaplanMeierFitter().fit(contrast.loc[contrast["tier_binary"] == 1, "SurvM"], contrast.loc[contrast["tier_binary"] == 1, "Event"], weights=weights[contrast["tier_binary"] == 1])
            rmst_control = rmst_from_kmf(km_control, tau)
            rmst_treat = rmst_from_kmf(km_treat, tau)
            rows.append({"contrast": f"{tier} vs POS=0", "tau_m": tau, "RMST_tier": rmst_treat, "RMST_POS0": rmst_control, "Delta_RMST": rmst_treat - rmst_control})
    return pd.DataFrame(rows)


def plot_km_by_group(ax: plt.Axes, df: pd.DataFrame, group_col: str, order: Sequence[str], title: str, ylabel: str = "Overall survival probability") -> None:
    kmf = KaplanMeierFitter()
    for label in order:
        sub = df.loc[df[group_col].astype(str) == label]
        if len(sub) < 10:
            continue
        kmf.fit(sub["SurvM"], sub["Event"], label=f"{label} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title(title)
    ax.set_xlabel("Months")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def format_logrank_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "Log-rank P = NA"
    if p_value < 0.001:
        return "Log-rank P < 0.001"
    return f"Log-rank P = {p_value:.3f}"


def km_groups_for_bjs_plot(df: pd.DataFrame, group_col: str, order: Sequence[str]) -> List[Tuple[str, pd.DataFrame]]:
    groups = []
    for label in order:
        sub = df.loc[df[group_col].astype(str) == label, ["SurvM", "Event"]].dropna().copy()
        if len(sub) < 10:
            continue
        groups.append((str(label), sub))
    return groups


def bjs_km_display_plan(
    groups: Sequence[Tuple[str, pd.DataFrame]],
    risk_times: Sequence[int],
    truncation_fraction: float,
) -> Dict[str, object]:
    starts = {label: int(len(sub)) for label, sub in groups}
    risk_counts = {
        label: {int(t): int((sub["SurvM"] >= float(t)).sum()) for t in risk_times}
        for label, sub in groups
    }

    display_times = []
    for t in risk_times:
        t_int = int(t)
        if all(risk_counts[label][t_int] >= starts[label] * truncation_fraction for label, _ in groups):
            display_times.append(t_int)
        else:
            break

    if not display_times:
        display_times = [int(risk_times[0])]
    max_time = int(display_times[-1])
    return {
        "display_times": display_times,
        "max_time": max_time,
        "risk_counts": risk_counts,
        "start_counts": starts,
    }


def draw_number_at_risk_table(
    risk_ax: plt.Axes,
    labels: Sequence[str],
    colors: Dict[str, str],
    display_times: Sequence[int],
    risk_counts: Dict[str, Dict[int, int]],
) -> None:
    risk_ax.set_axis_off()
    risk_ax.set_xlim(0, 1)
    risk_ax.set_ylim(0, 1)

    label_x = 0.02
    data_start = 0.40
    data_end = 0.98
    max_time = max(display_times) if display_times else 1
    if max_time <= 0:
        xpos = {int(t): data_start for t in display_times}
    else:
        xpos = {int(t): data_start + (float(t) / max_time) * (data_end - data_start) for t in display_times}

    risk_ax.text(label_x, 0.90, "Number at risk", transform=risk_ax.transAxes, ha="left", va="center", fontsize=8.5, fontweight="bold")
    for t in display_times:
        risk_ax.text(xpos[int(t)], 0.90, str(int(t)), transform=risk_ax.transAxes, ha="center", va="center", fontsize=8.5)

    n_rows = max(len(labels), 1)
    y_top = 0.70
    y_bottom = 0.14
    y_step = 0 if n_rows == 1 else (y_top - y_bottom) / (n_rows - 1)
    for i, label in enumerate(labels):
        y = y_top - i * y_step
        risk_ax.text(label_x, y, label, transform=risk_ax.transAxes, ha="left", va="center", fontsize=8, color=colors.get(label, "black"))
        for t in display_times:
            risk_ax.text(
                xpos[int(t)],
                y,
                str(int(risk_counts[label][int(t)])),
                transform=risk_ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )


def plot_km_by_group_bjs(
    ax: plt.Axes,
    risk_ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    order: Sequence[str],
    title: str,
    config: AnalysisConfig,
    ylabel: str = "Overall survival probability",
) -> Dict[str, object]:
    groups = km_groups_for_bjs_plot(df, group_col, order)
    plan = bjs_km_display_plan(groups, config.km_risk_times, config.km_truncation_fraction)
    display_times = plan["display_times"]
    max_time = int(plan["max_time"])

    kmf = KaplanMeierFitter()
    colors: Dict[str, str] = {}
    for label, sub in groups:
        kmf.fit(sub["SurvM"], sub["Event"], label=f"{label} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
        colors[label] = ax.get_lines()[-1].get_color()

    included = [label for label, _ in groups]
    test_df = df.loc[df[group_col].astype(str).isin(included), ["SurvM", "Event", group_col]].dropna().copy()
    if len(test_df) and test_df[group_col].astype(str).nunique() > 1:
        logrank = multivariate_logrank_test(test_df["SurvM"], test_df[group_col].astype(str), test_df["Event"])
        p_text = format_logrank_p(float(logrank.p_value))
        ax.text(0.02, 0.04, p_text, transform=ax.transAxes, fontsize=9)
        plan["logrank_p_value"] = float(logrank.p_value)
    else:
        plan["logrank_p_value"] = np.nan

    ax.set_title(title)
    ax.set_xlim(0, max_time)
    ax.set_xticks(display_times)
    ax.set_xlabel("Months", labelpad=3)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    draw_number_at_risk_table(
        risk_ax,
        included,
        colors,
        display_times,
        plan["risk_counts"],
    )
    plan["included_groups"] = included
    return plan


def create_figure_1(df: pd.DataFrame, overall_bins: pd.DataFrame, exact_curve: pd.DataFrame, config: AnalysisConfig, figure_path: Path) -> Dict[str, object]:
    fig = plt.figure(figsize=(13, 11.3), constrained_layout=True)
    fig._skip_tight_layout = True
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], wspace=0.24, hspace=0.26)
    ax1 = fig.add_subplot(gs[0, 0])
    km_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.04)
    ax2 = fig.add_subplot(km_gs[0, 0])
    risk_ax2 = fig.add_subplot(km_gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.hist(df["ELN"], bins=30, color="#4C78A8", alpha=0.75, edgecolor="white")
    ax1.axvline(df["ELN"].median(), color="#E45756", linestyle="--", linewidth=1.5, label=f"Median={df['ELN'].median():.0f}")
    ax1.set_title("Distribution of examined lymph nodes")
    ax1.set_xlabel("ELN")
    ax1.set_ylabel("Count")
    ax1.legend(frameon=False)

    d = df.copy()
    d["ELN_bin_plot"] = pd.cut(df["ELN"], bins=config.eln_bins_plot, labels=config.eln_bin_labels_plot)
    km_plan = plot_km_by_group_bjs(ax2, risk_ax2, d, "ELN_bin_plot", config.eln_bin_labels_plot, "Kaplan-Meier curves by ELN groups", config)

    bins_df = overall_bins.copy()
    ax3.plot(bins_df["ELN_bin"], bins_df["Median_OS_months"], marker="o", color="#59A14F", linewidth=2)
    for x, y, n in zip(bins_df["ELN_bin"], bins_df["Median_OS_months"], bins_df["n"]):
        ax3.annotate(f"{y:.0f}\n(n={n})", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax3.set_title("Median overall survival across detailed ELN bins")
    ax3.set_xlabel("ELN bin")
    ax3.set_ylabel("Median OS (months)")

    exact = exact_curve.copy()
    upper = int(np.nanquantile(exact["ELN"], config.exact_curve_quantile))
    exact = exact.loc[(exact["ELN"] <= upper) & (exact["n"] >= config.exact_curve_min_n)].copy()
    ax4.plot(exact["ELN"], exact["Median_OS_months"], color="#BAB0AC", linewidth=1.0, label="Raw")
    ax4.plot(exact["ELN"], exact["Median_OS_smoothed"], color="#F28E2B", linewidth=2.4, label="Smoothed")
    ax4.set_title(f"Smoothed median OS by exact ELN (n>={config.exact_curve_min_n} per ELN)")
    ax4.set_xlabel("ELN")
    ax4.set_ylabel("Median OS (months)")
    ax4.legend(frameon=False)

    for label, ax in zip(["A", "B", "C", "D"], [ax1, ax2, ax3, ax4]):
        add_panel_label(ax, label)
    save_figure(fig, figure_path, config.fig_dpi)
    return km_plan


def create_figure_2(threshold_df: pd.DataFrame, changepoint_df: pd.DataFrame, changepoint_cut: int, changepoint_ci: Tuple[float, float], eln_curve_df: pd.DataFrame, eln_reference: float, cox_summary: pd.DataFrame, config: AnalysisConfig, figure_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    best_cut = int(threshold_df.loc[threshold_df["logrank_stat"].idxmax(), "cut"])
    ax1.plot(threshold_df["cut"], threshold_df["logrank_stat"], marker="o", color="#4C78A8")
    ax1.axvline(best_cut, color="#E15759", linestyle="--", label=f"Best cut={best_cut}")
    ax1.axvline(config.primary_cut, color="#59A14F", linestyle=":", label=f"Primary cut={config.primary_cut}")
    ax1.set_title("Log-rank threshold profile for ELN")
    ax1.set_xlabel("Candidate ELN cut")
    ax1.set_ylabel("Log-rank statistic")
    ax1.legend(frameon=False)

    delta = changepoint_df["loglik"] - changepoint_df["loglik"].max()
    ax2.plot(changepoint_df["cut"], delta, marker="o", color="#B07AA1")
    ax2.axvline(changepoint_cut, color="#E15759", linestyle="--", label=f"Change point={changepoint_cut}")
    if np.isfinite(changepoint_ci[0]) and np.isfinite(changepoint_ci[1]):
        ax2.axvspan(changepoint_ci[0], changepoint_ci[1], color="#B07AA1", alpha=0.18, label="Bootstrap 95% CI")
    ax2.set_title("Segmented Cox change-point profile")
    ax2.set_xlabel("Candidate ELN cut")
    ax2.set_ylabel("Log-likelihood difference")
    ax2.legend(frameon=False)

    ax3.plot(eln_curve_df["ELN"], eln_curve_df["HR"], color="#F28E2B", linewidth=2)
    ax3.fill_between(eln_curve_df["ELN"], eln_curve_df["HR_lower"], eln_curve_df["HR_upper"], color="#F28E2B", alpha=0.20)
    ax3.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax3.axvline(eln_reference, color="grey", linestyle=":", linewidth=1)
    ax3.set_title("Adjusted spline curve for ELN")
    ax3.set_xlabel("ELN")
    ax3.set_ylabel("Relative hazard")

    forest = cox_summary.copy()
    forest = forest.loc[~forest["pretty_term"].str.contains("Other|Unknown", na=False)]
    forest = forest.sort_values("exp(coef)")
    y = np.arange(len(forest))
    hr = forest["exp(coef)"].to_numpy()
    lo = forest["exp(coef) lower 95%"].to_numpy()
    hi = forest["exp(coef) upper 95%"].to_numpy()
    ax4.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", color="#4C78A8", capsize=3)
    ax4.axvline(1.0, color="grey", linestyle="--")
    ax4.set_yticks(y)
    ax4.set_yticklabels(forest["pretty_term"])
    ax4.set_xlabel("Hazard ratio")
    ax4.set_title("Multivariable Cox model")

    for label, ax in zip(["A", "B", "C", "D"], axes.flatten()):
        add_panel_label(ax, label)
    save_figure(fig, figure_path, config.fig_dpi)


def create_figure_3(calibration_tables: Dict[int, pd.DataFrame], dca_tables: Dict[int, pd.DataFrame], config: AnalysisConfig, figure_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    labels = ["A", "B", "C", "D", "E", "F"]

    for i, tau in enumerate(config.calibration_times):
        ax = axes[0, i]
        tab = calibration_tables[tau]
        ax.plot(tab["predicted"], tab["observed"], marker="o", color="#4C78A8")
        lim = max(float(tab["predicted"].max()), float(tab["observed"].max()), 0.01)
        ax.plot([0, lim], [0, lim], linestyle="--", color="grey")
        ax.set_title(f"Calibration at {tau} months")
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Observed risk")

    for i, tau in enumerate(config.calibration_times):
        ax = axes[1, i]
        tab = dca_tables[tau]
        ax.plot(tab["threshold"], tab["net_benefit_model"], label="Model", color="#E15759", linewidth=2)
        ax.plot(tab["threshold"], tab["net_benefit_all"], label="Treat all", color="#59A14F", linestyle="--")
        ax.plot(tab["threshold"], tab["net_benefit_none"], label="Treat none", color="grey", linestyle=":")
        ax.set_title(f"DCA at {tau} months")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        if i == 0:
            ax.legend(frameon=False)

    for label, ax in zip(labels, axes.flatten()):
        add_panel_label(ax, label)
    save_figure(fig, figure_path, config.fig_dpi)


def create_figure_4(df: pd.DataFrame, pos_transition: pd.DataFrame, pos_curve: pd.DataFrame, pos_rmst: pd.DataFrame, config: AnalysisConfig, figure_path: Path) -> Dict[str, object]:
    fig = plt.figure(figsize=(13, 11.3), constrained_layout=True)
    fig._skip_tight_layout = True
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], wspace=0.24, hspace=0.26)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    km_gs = gs[1, 0].subgridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.04)
    ax3 = fig.add_subplot(km_gs[0, 0])
    risk_ax3 = fig.add_subplot(km_gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(pos_transition))
    ax1.bar(x, pos_transition["HR"], color=["#BAB0AC", "#4C78A8"], width=0.6)
    ax1.errorbar(x, pos_transition["HR"], yerr=[pos_transition["HR"] - pos_transition["HR_lower"], pos_transition["HR_upper"] - pos_transition["HR"]], fmt="none", ecolor="black", capsize=4)
    ax1.axhline(1.0, color="grey", linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pos_transition["group"])
    ax1.set_ylabel("Relative hazard")
    ax1.set_title("Step effect of positive lymph nodes")

    ax2.plot(pos_curve["POS"], pos_curve["HR"], color="#E15759", linewidth=2)
    ax2.fill_between(pos_curve["POS"], pos_curve["HR_lower"], pos_curve["HR_upper"], color="#E15759", alpha=0.20)
    ax2.axhline(1.0, color="grey", linestyle="--")
    ax2.set_xlabel("POS")
    ax2.set_ylabel("Relative hazard")
    ax2.set_title("Adjusted spline curve for POS")

    d = df.copy()
    d["POS_tier"] = d["POS"].map(pos_tier)
    km_plan = plot_km_by_group_bjs(ax3, risk_ax3, d, "POS_tier", ["POS=0", "POS=1-2", "POS=3-4", "POS>=5"], "Kaplan-Meier curves by POS tier", config)

    heat = pos_rmst.pivot(index="contrast", columns="tau_m", values="Delta_RMST").reindex(["POS=1-2 vs POS=0", "POS=3-4 vs POS=0", "POS>=5 vs POS=0"])
    im = ax4.imshow(heat.to_numpy(dtype=float), cmap="RdBu_r", aspect="auto")
    ax4.set_xticks(np.arange(len(heat.columns)))
    ax4.set_xticklabels([f"{c}m" for c in heat.columns])
    ax4.set_yticks(np.arange(len(heat.index)))
    ax4.set_yticklabels(heat.index)
    ax4.set_title("Weighted Delta RMST by POS contrast")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax4.text(j, i, f"{heat.iloc[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="Delta RMST (months)")

    for label, ax in zip(["A", "B", "C", "D"], [ax1, ax2, ax3, ax4]):
        add_panel_label(ax, label)
    save_figure(fig, figure_path, config.fig_dpi)
    return km_plan


def create_supplementary_stage_figure(df: pd.DataFrame, config: AnalysisConfig, figure_path: Path) -> None:
    d = df.loc[df["Stage"].astype(str).isin(["Localized", "Regional", "Distant"])].copy()
    d["ELN_bin_plot"] = pd.cut(d["ELN"], bins=config.eln_bins_plot, labels=config.eln_bin_labels_plot)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for idx, stage in enumerate(["Localized", "Regional", "Distant"]):
        sub = d.loc[d["Stage"].astype(str) == stage]
        plot_km_by_group(axes[idx], sub, "ELN_bin_plot", config.eln_bin_labels_plot, f"{stage} stage")
        add_panel_label(axes[idx], chr(ord("A") + idx))
    save_figure(fig, figure_path, config.fig_dpi)


def create_supplementary_rmst_figure(
    strategy_rmst: pd.DataFrame,
    strategy_balance: pd.DataFrame,
    chosen_method: str,
    config: AnalysisConfig,
    figure_path: Path,
) -> None:
    plot_df = strategy_rmst.loc[strategy_rmst["method"].astype(str).str.upper() == chosen_method.upper()].copy()
    plot_df = plot_df.sort_values(["cut", "tau_m"])
    balance_df = (
        strategy_balance.loc[strategy_balance["method"].astype(str).str.upper() == chosen_method.upper()]
        .assign(abs_after=lambda d: d["SMD_after"].abs())
        .groupby("cut", as_index=False)["abs_after"]
        .mean()
        .rename(columns={"abs_after": "mean_abs_smd"})
    )

    palette = {
        cut: color
        for cut, color in zip(
            sorted(plot_df["cut"].unique().tolist()),
            ["#2F6690", "#D17C00", "#1E7F5C", "#B85C38"],
        )
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax1, ax2 = axes

    cuts = sorted(plot_df["cut"].unique().tolist())
    taus = list(config.calibration_times)
    x = np.arange(len(taus))
    width = 0.24

    for idx, cut in enumerate(cuts):
        sub = plot_df.loc[plot_df["cut"] == cut].set_index("tau_m").reindex(taus).reset_index()
        xpos = x + (idx - (len(cuts) - 1) / 2) * width
        bars = ax1.bar(
            xpos,
            sub["Delta_RMST"],
            width=width,
            color=palette[cut],
            edgecolor="black",
            linewidth=0.6,
            label=f"ELN≥{cut}",
            alpha=0.95,
        )
        for bar, value in zip(bars, sub["Delta_RMST"]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.1,
            )

    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{tau} m" for tau in taus])
    ax1.set_xlabel("RMST horizon")
    ax1.set_ylabel("Delta RMST (months)")
    ax1.set_title("Overlap-weighted RMST gain by ELN target")
    ax1.legend(frameon=False, ncol=len(cuts), loc="upper left")
    ax1.set_ylim(0, max(0.8, float(plot_df["Delta_RMST"].max()) + 0.55))

    wide = plot_df.pivot(index="cut", columns="tau_m", values="Delta_RMST").reindex(cuts)
    im = ax2.imshow(wide.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto", vmin=float(wide.min().min()) - 0.1, vmax=float(wide.max().max()) + 0.1)
    ax2.set_xticks(np.arange(len(wide.columns)))
    ax2.set_xticklabels([f"{int(c)} m" for c in wide.columns])
    ax2.set_yticks(np.arange(len(wide.index)))
    ax2.set_yticklabels([f"ELN≥{int(idx)}" for idx in wide.index])
    ax2.set_title(f"{chosen_method} weighted Delta RMST")
    for i in range(len(wide.index)):
        for j in range(len(wide.columns)):
            ax2.text(j, i, f"{wide.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8.8, color="#10233A", fontweight="bold")

    ax2b = ax2.inset_axes([1.06, 0.10, 0.06, 0.80])
    cbar = fig.colorbar(im, cax=ax2b)
    cbar.set_label("Delta RMST")

    add_panel_label(ax1, "A")
    add_panel_label(ax2, "B")
    save_figure(fig, figure_path, config.fig_dpi)


def write_tables(workbook_path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(workbook_path) as writer:
        for sheet_name, df in sheets.items():
            public_name = PART1_SHEET_NAME_MAP.get(sheet_name, sheet_name)
            df.to_excel(writer, sheet_name=sanitize_sheet_name(public_name), index=False)


def write_manifest(path: Path, figure_dir: Path, workbook_path: Path, summary: Dict[str, object], sheets: Dict[str, pd.DataFrame]) -> None:
    lines = ["Curated manuscript outputs", "", "Figures:"]
    for item in sorted(figure_dir.glob("*.png")):
        lines.append(f"- {item.name}")
    lines.extend(["", "Workbook:", f"- {workbook_path.name}", "", "Workbook sheets:"])
    for name in sheets:
        lines.append(f"- {sanitize_sheet_name(name)}")
    lines.extend(["", "Summary:", json.dumps(summary, ensure_ascii=False, indent=2)])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manuscript-ready merged ELN/POS pipeline")
    parser.add_argument("--excel", default="Pancreatic_Body_Tail_ALL.xlsx", help="Input Excel file")
    parser.add_argument("--sheet", default="export", help="Worksheet name")
    parser.add_argument("--outdir", default="outputs_part1_seer_analysis_v3", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Use reduced bootstrap counts for faster verification")
    args = parser.parse_args()

    config = AnalysisConfig(excel_path=args.excel, sheet_name=args.sheet, outdir=args.outdir)
    if args.quick:
        config.cindex_bootstrap_n = 40
        config.changepoint_bootstrap_n = 40
        config.dca_points = 40

    set_publication_style()

    outdir = Path(config.outdir)
    figure_dir = outdir / "figures"
    table_dir = outdir / "tables"
    ensure_dir(figure_dir)
    ensure_dir(table_dir)

    print("Reading and cleaning dataset ...")
    analysis_df, cohort_summary = load_and_prepare_dataset(config)

    print("Building descriptive summaries ...")
    eln_overall, eln_by_stage, eln_by_surgery, eln_exact = summarize_eln_bins(analysis_df, config)
    threshold_df = threshold_search(analysis_df, config)
    best_threshold = int(threshold_df.loc[threshold_df["logrank_stat"].idxmax(), "cut"]) if not threshold_df.empty else np.nan
    baseline = build_baseline_table(analysis_df, best_threshold)
    descriptive_table = build_manuscript_descriptive_table(analysis_df, best_threshold)
    changepoint_df, changepoint_cut, changepoint_ci = changepoint_search(analysis_df, config)

    print("Fitting main ELN models ...")
    cph_main, cox_fit_data, cox_summary, ph_summary, cindex_df = fit_main_cox(analysis_df, config)
    cindex_mean = float(cindex_df["Bootstrap_Cindex"].mean()) if not cindex_df.empty else np.nan
    cindex_ci = tuple(np.percentile(cindex_df["Bootstrap_Cindex"], [2.5, 97.5]).tolist()) if len(cindex_df) >= 10 else (np.nan, np.nan)

    spline_cph, _, spline_design = build_eln_spline_model(analysis_df)
    spline_ref = pd.DataFrame(
        [
            {
                "ELN": float(analysis_df["ELN"].median()),
                "Age_per10": float(analysis_df["Age_per10"].median()),
                "chemo_bin": int(analysis_df["chemo_bin"].mode().iloc[0]),
                "radiation_bin": int(analysis_df["radiation_bin"].mode().iloc[0]),
                "Sex_clean": str(analysis_df["Sex_clean"].mode().iloc[0]),
                "Stage": str(analysis_df["Stage"].mode().iloc[0]),
                "Primary_part": str(analysis_df["Primary_part"].mode().iloc[0]),
                "SurgeryExtent_clean": str(analysis_df["SurgeryExtent_clean"].mode().iloc[0]),
            }
        ]
    )
    eln_grid = np.linspace(1, float(np.nanquantile(analysis_df["ELN"], 0.99)), 120)
    eln_curve_df = predict_relative_hr_curve(spline_cph, spline_design, spline_ref, "ELN", eln_grid, float(analysis_df["ELN"].median()))

    print("Generating calibration and DCA ...")
    calibration_tables, dca_tables = generate_calibration_and_dca(cph_main, cox_fit_data, config)

    print("Running strategy summaries ...")
    strategy_rmst, strategy_balance = run_strategy_analysis(analysis_df, config)
    chosen_method = select_preferred_weight_method(strategy_balance)
    strategy_rmst_main = strategy_rmst.loc[strategy_rmst["method"].astype(str).str.upper() == chosen_method.upper()].reset_index(drop=True)

    print("Fitting POS model ...")
    pos_cph, pos_fit_data, pos_design, pos_formula = build_pos_model(analysis_df)
    pos_summary = pos_cph.summary.reset_index().rename(columns={"index": "covariate"})
    pos_summary["pretty_term"] = pos_summary["covariate"].map(pretty_term)
    pos_ph = proportional_hazard_test(pos_cph, pos_fit_data, time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
    pos_ph["pretty_term"] = pos_ph["covariate"].map(pretty_term)
    pos_reference = build_pos_reference_profile(analysis_df)
    quantiles = analysis_df.loc[analysis_df["POS"] > 0, "POS"].quantile([0.25, 0.5, 0.75, 0.9]).round().astype(int).tolist()
    pos_grid = sorted(set([1, 2, 3, 5, 8, 10] + quantiles))
    pos_grid = [value for value in pos_grid if 1 <= value <= int(analysis_df["POS"].max())]
    pos_transition, pos_curve = predict_pos_effect_curve(pos_cph, pos_design, pos_reference, pos_grid)
    pos_rmst = compute_pos_weighted_rmst(analysis_df)

    print("Creating curated figures ...")
    figure1_km_plan = create_figure_1(analysis_df, eln_overall, eln_exact, config, figure_dir / "Figure_1_SEER_ELN_Overview.png")
    create_figure_2(threshold_df, changepoint_df, changepoint_cut, changepoint_ci, eln_curve_df, float(analysis_df["ELN"].median()), cox_summary, config, figure_dir / "Figure_2_SEER_ELN_Modeling.png")
    create_figure_3(calibration_tables, dca_tables, config, figure_dir / "Supplementary_Figure_S1_SEER_Model_Performance.png")
    figure3_km_plan = create_figure_4(analysis_df, pos_transition, pos_curve, pos_rmst, config, figure_dir / "Figure_3_SEER_POS_Analysis.png")
    create_supplementary_stage_figure(analysis_df, config, figure_dir / "Supplementary_Figure_S2_SEER_ELN_By_Stage.png")
    create_supplementary_rmst_figure(strategy_rmst_main, strategy_balance, chosen_method, config, figure_dir / "Supplementary_Figure_S3_SEER_RMST_OW.png")
    print(f"[V3] Figure 1B KM truncated at {figure1_km_plan['max_time']} months; risk table times: {figure1_km_plan['display_times']}")
    print(f"[V3] Figure 3C KM truncated at {figure3_km_plan['max_time']} months; risk table times: {figure3_km_plan['display_times']}")

    summary_sheet = pd.DataFrame(
        [
            {"Item": "Raw cohort", "Value": cohort_summary["raw_n"]},
            {"Item": "Removed POS>ELN", "Value": cohort_summary["removed_pos_gt_eln"]},
            {"Item": "Analysis cohort", "Value": cohort_summary["analysis_n"]},
            {"Item": "Events", "Value": cohort_summary["event_n"]},
            {"Item": "Censored", "Value": cohort_summary["censored_n"]},
            {"Item": "Stage unknown", "Value": cohort_summary["stage_unknown_n"]},
            {"Item": "Primary ELN cut", "Value": config.primary_cut},
            {"Item": "Best log-rank cut", "Value": best_threshold},
            {"Item": "Change point", "Value": changepoint_cut},
            {"Item": "Change point 95% CI lower", "Value": changepoint_ci[0]},
            {"Item": "Change point 95% CI upper", "Value": changepoint_ci[1]},
            {"Item": "Bootstrap c-index mean", "Value": cindex_mean},
            {"Item": "Bootstrap c-index 95% CI lower", "Value": cindex_ci[0]},
            {"Item": "Bootstrap c-index 95% CI upper", "Value": cindex_ci[1]},
            {"Item": "Preferred RMST weighting", "Value": chosen_method},
            {"Item": "POS model formula", "Value": pos_formula},
        ]
    )

    sheets = {
        "Summary": summary_sheet,
        "Descriptive_Stats": descriptive_table,
        "Baseline": baseline,
        "ELN_Bins_Overall": eln_overall,
        "ELN_Bins_Stage": eln_by_stage,
        "ELN_Bins_Surgery": eln_by_surgery,
        "ELN_Exact_Curve": eln_exact,
        "ELN_Threshold": threshold_df,
        "ELN_Changepoint": changepoint_df,
        "ELN_Cox": cox_summary,
        "ELN_PH": ph_summary,
        "ELN_Cindex": cindex_df,
        "ELN_Strategy_RMST": strategy_rmst_main,
        "ELN_Strategy_RMST_AllMethods": strategy_rmst,
        "ELN_Balance": strategy_balance,
        "ELN_Spline_Curve": eln_curve_df,
        "ELN_Calib_12m": calibration_tables[12],
        "ELN_Calib_36m": calibration_tables[36],
        "ELN_Calib_60m": calibration_tables[60],
        "ELN_DCA_12m": dca_tables[12],
        "ELN_DCA_36m": dca_tables[36],
        "ELN_DCA_60m": dca_tables[60],
        "POS_Cox": pos_summary,
        "POS_PH": pos_ph,
        "POS_Transition": pos_transition,
        "POS_Curve": pos_curve,
        "POS_RMST": pos_rmst,
    }

    workbook_path = table_dir / "Part1_SEER_Tables.xlsx"
    print("Writing workbook ...")
    write_tables(workbook_path, sheets)

    manifest_summary = {
        **cohort_summary,
        "best_threshold": best_threshold,
        "changepoint_cut": changepoint_cut,
        "changepoint_ci": changepoint_ci,
        "bootstrap_cindex_mean": cindex_mean,
        "bootstrap_cindex_ci": cindex_ci,
        "preferred_rmst_weighting": chosen_method,
        "quick_mode": args.quick,
    }
    write_manifest(outdir / "Output_Index.txt", figure_dir, workbook_path, manifest_summary, sheets)
    km_metadata = {
        "Figure_1B_ELN_KM": figure1_km_plan,
        "Figure_3C_POS_KM": figure3_km_plan,
        "rule": {
            "risk_times_months": list(config.km_risk_times),
            "truncation_fraction_of_start": config.km_truncation_fraction,
            "at_risk_definition": "Count of original survival records with SurvM >= displayed month within each plotted group.",
        },
    }
    with open(outdir / "V3_KM_Risk_Table_Metadata.json", "w", encoding="utf-8") as fh:
        json.dump(km_metadata, fh, indent=2)
    print(f"Completed. Curated outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
