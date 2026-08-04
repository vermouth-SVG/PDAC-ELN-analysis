"""
Part 2 anatomical-evidence pipeline.

The current revision is built around one question:
which surgical approach is more likely to move patients toward the Part 1
high-yield anchor, especially ELN >= 21?
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.colors import ListedColormap
from matplotlib.transforms import blended_transform_factory


NORM = NormalDist()


@dataclass
class Config:
    outdir: Path = Path("outputs_part2_anatomical_evidence")
    dpi: int = 300
    thresholds: tuple[int, int, int] = (12, 14, 21)
    part1_tables: Path = Path("outputs_part1_seer_analysis") / "tables" / "Part1_SEER_Tables.xlsx"
    part1_main_tables: Path = Path("outputs_part1_seer_analysis") / "main_text_tables" / "Part1_Main_Text_Tables.xlsx"


PART2_SHEET_NAME_MAP = {
    "claim_audit": "Claim_Audit",
    "study_catalog": "Study_Catalog",
    "reference_summary": "Reference_Summary",
    "usage_map": "Literature_Usage_Map",
    "fig4_source": "Figure4_Harmonized_Cohorts",
    "fig5_access": "Figure5_Station_Access",
    "fig6_scores": "SuppFigureS4_Heterogeneity",
    "fig6_rationale": "SuppFigureS4_Heterogeneity_Legend",
    "fig10_ramps": "Figure6_Anchor_Probabilities",
    "fig10_source_basis": "Figure6_Family_Distribution_Summ",
    "fig12_quant": "Figure7_Anchor_Association",
    "fig12_qual": "Figure7_Association_Notes",
    "table3_approaches": "Table3_Approach_Comparison",
    "table4_access": "Table4_Anatomical_Access",
    "table5_threshold": "Table5_Anchor_Attainment",
    "table6_deep": "Table6_Reference_Changes",
    "method_notes": "Method_Traceability_Notes",
    "stat_validation": "Statistical_Validation",
    "recommended_refs": "Recommended_References",
}

PART2_MAIN_FIGURES = [
    "Figure_4_Approach_ELN_Landscape",
    "Figure_5_Station_Accessibility",
    "Figure_6_RAMPS_MultiStudy_Forest",
    "Figure_7_Threshold_Evidence_Map",
]

PART2_SUPPLEMENTARY_FIGURES = [
    "Supplementary_Figure_S4_Approach_Profile",
    "Supplementary_Figure_S5_Anchor_Attainment_Matrix",
]

PART2_MAIN_TABLE_SHEETS = [
    "Table3_Approach_Comparison",
    "Table4_Anatomical_Access",
    "Table5_Anchor_Attainment",
]


def sanitize_sheet_name(name: str) -> str:
    for token in ["[", "]", ":", "*", "?", "/", "\\"]:
        name = name.replace(token, "_")
    return name[:31]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
            "axes.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_part1_anchors(config: Config) -> dict[str, float]:
    anchors = {
        "best_cut": 12.0,
        "hr1_cross": 14.0,
        "hr1_display": 14.0,
        "change_point": 21.0,
        "rmst_12_60": 2.586834,
        "rmst_14_60": 2.307580,
        "rmst_21_60": 2.601004,
    }
    if config.part1_tables.exists():
        try:
            xl = pd.ExcelFile(config.part1_tables)
            summary_sheet = "Study_Summary" if "Study_Summary" in xl.sheet_names else "Summary"
            spline_sheet = "ELN_Spline_HR_Curve" if "ELN_Spline_HR_Curve" in xl.sheet_names else "ELN_Spline_Curve"
            summary = pd.read_excel(config.part1_tables, sheet_name=summary_sheet)
            mapping = dict(zip(summary["Item"], summary["Value"]))
            anchors["best_cut"] = float(mapping.get("Best log-rank cut", anchors["best_cut"]))
            anchors["change_point"] = float(mapping.get("Change point", anchors["change_point"]))
            spline = pd.read_excel(config.part1_tables, sheet_name=spline_sheet).dropna(subset=["ELN", "HR"]).sort_values("ELN").reset_index(drop=True)
            for i in range(len(spline) - 1):
                x1, y1 = float(spline.loc[i, "ELN"]), float(spline.loc[i, "HR"])
                x2, y2 = float(spline.loc[i + 1, "ELN"]), float(spline.loc[i + 1, "HR"])
                if y1 == 1.0:
                    anchors["hr1_cross"] = x1
                    break
                if (y1 - 1.0) * (y2 - 1.0) < 0:
                    anchors["hr1_cross"] = x1 + (1.0 - y1) * (x2 - x1) / (y2 - y1)
                    break
            anchors["hr1_display"] = float(int(round(anchors["hr1_cross"])))
        except Exception:
            pass
    if config.part1_main_tables.exists():
        try:
            rmst = pd.read_excel(config.part1_main_tables, sheet_name="Table2_SEER_RMST")
            for cut in [12, 14, 21]:
                match = rmst[(rmst["cut"] == cut) & (rmst["tau_m"] == 60)]
                if not match.empty:
                    anchors[f"rmst_{cut}_60"] = float(match["Delta_RMST"].iloc[0])
        except Exception:
            pass
    return anchors


def estimate_mean_sd(row: pd.Series) -> tuple[float, float, str]:
    stat_type = row["summary_type"]
    n = int(row["n"])
    center = float(row["center"])
    low = float(row["low"]) if pd.notna(row["low"]) else np.nan
    high = float(row["high"]) if pd.notna(row["high"]) else np.nan
    if stat_type == "mean_sd":
        return center, (high - low) / 2.0, "reported mean±SD"
    if stat_type == "median_iqr":
        return center, (high - low) / 1.349, "median/IQR to SD-equivalent"
    if stat_type == "median_range":
        z = NORM.inv_cdf((n - 0.375) / (n + 0.25)) if n > 1 else np.nan
        mean = center if n > 70 else (low + 2 * center + high) / 4.0
        sd = (high - low) / (2 * z) if np.isfinite(z) and z > 0 else np.nan
        return mean, sd, "median/range to SD-equivalent"
    return np.nan, np.nan, "not estimable"


def prob_ge_threshold(mean: float, sd: float, threshold: int) -> float:
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        return np.nan
    # Use a continuity correction because ELN is a discrete count.
    cutoff = threshold - 0.5
    return float(max(0.0, min(1.0, 1.0 - NormalDist(mu=mean, sigma=sd).cdf(cutoff))))


def ci_from_probability(prob: float, n: int) -> tuple[float, float]:
    if not np.isfinite(prob) or n <= 0:
        return np.nan, np.nan
    se = np.sqrt(prob * max(1 - prob, 1e-9) / n)
    low = max(0.0, prob - 1.96 * se)
    high = min(1.0, prob + 1.96 * se)
    return low, high


def combine_mean_sd_within_group(sub: pd.DataFrame) -> dict[str, object]:
    estimable = sub[sub["eligible_for_probability"]].copy()
    quantitative_papers = int(sub["study"].nunique())
    estimable_studies = int(estimable["study"].nunique())
    if estimable.empty:
        return {
            "estimable_cohorts": 0,
            "combined_n": 0,
            "quantitative_papers": quantitative_papers,
            "estimable_studies": estimable_studies,
            "combined_mean": np.nan,
            "combined_sd": np.nan,
            "combined_mean_low": np.nan,
            "combined_mean_high": np.nan,
        }

    weights = estimable["n"].to_numpy(dtype=float)
    means = estimable["mean_equiv"].to_numpy(dtype=float)
    sds = estimable["sd_equiv"].to_numpy(dtype=float)
    total_n = int(weights.sum())
    combined_mean = float(np.average(means, weights=weights))
    if total_n > 1:
        pooled_ss = float(np.sum((weights - 1.0) * (sds**2) + weights * ((means - combined_mean) ** 2)))
        combined_sd = float(np.sqrt(max(pooled_ss / (total_n - 1), 1e-9)))
    else:
        combined_sd = float(sds[0]) if len(sds) else np.nan
    mean_se = combined_sd / np.sqrt(total_n) if np.isfinite(combined_sd) and total_n > 0 else np.nan
    return {
        "estimable_cohorts": int(len(estimable)),
        "combined_n": total_n,
        "quantitative_papers": quantitative_papers,
        "estimable_studies": estimable_studies,
        "combined_mean": combined_mean,
        "combined_sd": combined_sd,
        "combined_mean_low": combined_mean - 1.96 * mean_se if np.isfinite(mean_se) else np.nan,
        "combined_mean_high": combined_mean + 1.96 * mean_se if np.isfinite(mean_se) else np.nan,
    }


def summarize_family_distributions(df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, sub in df.groupby("family", sort=False):
        out: dict[str, object] = {"family": family}
        out.update(combine_mean_sd_within_group(sub))
        out["formal_testable"] = (
            out["estimable_cohorts"] >= 3
            and out["estimable_studies"] >= 3
            and family != "Posterior / artery-first spectrum"
            and np.isfinite(out["combined_mean"])
            and np.isfinite(out["combined_sd"])
        )
        for thr in thresholds:
            prob = prob_ge_threshold(out["combined_mean"], out["combined_sd"], thr)
            low, high = ci_from_probability(prob, int(out["combined_n"]))
            out[f"p_ge_{thr}"] = prob
            out[f"p_ge_{thr}_low"] = low
            out[f"p_ge_{thr}_high"] = high
        rows.append(out)
    return pd.DataFrame(rows)


def pairwise_family_tests(family_pool: pd.DataFrame) -> pd.DataFrame:
    tests: list[dict[str, object]] = []
    testable = family_pool[family_pool["formal_testable"]].copy()
    for left, right in combinations(testable["family"].tolist(), 2):
        lrow = testable[testable["family"] == left].iloc[0]
        rrow = testable[testable["family"] == right].iloc[0]
        for thr in [12, 14, 21]:
            lp = float(lrow[f"p_ge_{thr}"])
            rp = float(rrow[f"p_ge_{thr}"])
            ln = max(int(lrow["combined_n"]), 1)
            rn = max(int(rrow["combined_n"]), 1)
            lse = np.sqrt(lp * (1 - lp) / ln)
            rse = np.sqrt(rp * (1 - rp) / rn)
            se = np.sqrt(lse**2 + rse**2)
            rd = lp - rp
            z = rd / se if se > 0 else np.nan
            pval = 2 * (1 - NORM.cdf(abs(z))) if np.isfinite(z) else np.nan
            rr = lp / rp if np.isfinite(lp) and np.isfinite(rp) and rp > 0 else np.nan
            left_events = max(lp * ln, 1e-9)
            right_events = max(rp * rn, 1e-9)
            rr_se = (
                np.sqrt(max((1 / left_events) - (1 / ln), 0.0) + max((1 / right_events) - (1 / rn), 0.0))
                if np.isfinite(rr) and rr > 0
                else np.nan
            )
            rr_low = np.exp(np.log(rr) - 1.96 * rr_se) if np.isfinite(rr) and np.isfinite(rr_se) else np.nan
            rr_high = np.exp(np.log(rr) + 1.96 * rr_se) if np.isfinite(rr) and np.isfinite(rr_se) else np.nan
            tests.append(
                {
                    "comparison": f"{left} vs {right}",
                    "threshold": f"ELN>={thr}",
                    "left_combined_mean": float(lrow["combined_mean"]),
                    "right_combined_mean": float(rrow["combined_mean"]),
                    "left_probability": lp,
                    "right_probability": rp,
                    "risk_difference": rd,
                    "risk_ratio": rr,
                    "rr_low": rr_low,
                    "rr_high": rr_high,
                    "z_value": z,
                    "approx_p_value": pval,
                    "note": "Approximate threshold comparison derived after first combining mean/SD within each family. RR and p-values are based on distribution-derived expected counts, not weighted averaging of study-specific probabilities.",
                }
            )
    if not tests:
        return pd.DataFrame(
            columns=[
                "comparison",
                "threshold",
                "left_combined_mean",
                "right_combined_mean",
                "left_probability",
                "right_probability",
                "risk_difference",
                "risk_ratio",
                "rr_low",
                "rr_high",
                "z_value",
                "approx_p_value",
                "note",
            ]
        )
    return pd.DataFrame(tests)


def pct_text(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value * 100:.1f}%"


def prob_ci_text(prob: float, low: float, high: float) -> str:
    if not np.isfinite(prob):
        return "NA"
    if np.isfinite(low) and np.isfinite(high):
        return f"{prob * 100:.1f}% ({low * 100:.1f}-{high * 100:.1f})"
    return f"{prob * 100:.1f}%"


def format_mean_sd(mean: float, sd: float) -> str:
    if not np.isfinite(mean) or not np.isfinite(sd):
        return "NA"
    return f"{mean:.1f}±{sd:.1f}"


def family_display_name(name: str) -> str:
    mapping = {
        "Standard / conventional DP": "Standard DP",
        "RAMPS": "RAMPS",
        "Posterior / artery-first spectrum": "Posterior spectrum",
    }
    return mapping.get(name, name)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")


def anchor_position_text(mean: float, threshold: float) -> str:
    if not np.isfinite(mean):
        return "not estimable"
    if mean >= threshold:
        return f"combined mean is at/above ELN={int(threshold)}"
    return f"combined mean remains below ELN={int(threshold)}"


def build_data(config: Config, anchors: dict[str, float]) -> dict[str, pd.DataFrame]:
    family_std = "Standard / conventional DP"
    family_ramps = "RAMPS"
    family_deep = "Posterior / artery-first spectrum"
    palette = {
        family_std: "#2F6690",
        family_ramps: "#D17C00",
        family_deep: "#1E7F5C",
    }
    family_order = {family_std: 1, family_ramps: 2, family_deep: 3}

    study_catalog = pd.DataFrame(
        [
            {
                "study": "Abe 2016",
                "year": 2016,
                "pmid": 27138881,
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "design": "Retrospective comparative cohort",
                "quantitative_role": "Estimable RAMPS cohort",
                "used_for_probability": "Yes",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "RAMPS 28.4±11.6 versus SRPS 20.7±10.1 nodes; R0 improved, OS not clearly superior.",
                "caution": "Single-center retrospective cohort.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/27138881/",
            },
            {
                "study": "Kim 2016",
                "year": 2016,
                "pmid": 27433462,
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "design": "Retrospective comparative cohort",
                "quantitative_role": "Estimable RAMPS cohort",
                "used_for_probability": "Yes",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "RAMPS harvested 21.5±8.3 nodes and improved tangential-margin negativity, with neutral OS.",
                "caution": "Small retrospective series.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/27433462/",
            },
            {
                "study": "Takaori 2016",
                "year": 2016,
                "pmid": 27215565,
                "family": family_deep,
                "subgroup": "Artery-first concept",
                "design": "Technical concept paper",
                "quantitative_role": "No direct ELN distribution",
                "used_for_probability": "No",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "No",
                "main_message": "Treitz-side artery-first DP explains why posterior and SMA-adjacent access may improve.",
                "caution": "Mechanistic support only; no direct ELN benchmark.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/27215565/",
            },
            {
                "study": "Kim 2017",
                "year": 2017,
                "pmid": 28061895,
                "family": family_ramps,
                "subgroup": "Minimally invasive RAMPS",
                "design": "Single-arm laparoscopic RAMPS series",
                "quantitative_role": "Estimable minimally invasive RAMPS cohort",
                "used_for_probability": "Yes",
                "used_for_access_map": "No",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "Initial laparoscopic RAMPS reported 18.1±6.2 nodes with negative margins in all patients.",
                "caution": "Selected early series without a direct comparator.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/28061895/",
            },
            {
                "study": "Ome 2019",
                "year": 2019,
                "pmid": 31502021,
                "family": family_deep,
                "subgroup": "Artery-first technical variant",
                "design": "Technical feasibility series",
                "quantitative_role": "No direct ELN distribution",
                "used_for_probability": "No",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "No",
                "main_message": "Caudo-dorsal artery-first LDP directly exposes the left SMA side and posterior plane early.",
                "caution": "Five-patient technical series without a comparable ELN benchmark.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/31502021/",
            },
            {
                "study": "Sham 2020",
                "year": 2020,
                "pmid": 32576018,
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "design": "Dual-institution comparative cohort",
                "quantitative_role": "Point-only counterexample",
                "used_for_probability": "No",
                "used_for_access_map": "No",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "DPS harvested more total nodes than RAMPS (18 versus 12); OS remained similar.",
                "caution": "Important heterogeneity signal; cannot estimate threshold probabilities from a point-only summary.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/32576018/",
            },
            {
                "study": "Nagai 2020",
                "year": 2020,
                "pmid": 33155070,
                "family": family_deep,
                "subgroup": "Left-posterior artery-first",
                "design": "Single-arm posterior-route series",
                "quantitative_role": "Estimable trend-support cohort",
                "used_for_probability": "Yes",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "Left-posterior artery-first LDP reported 35 (11-49) nodes with explicit deep posterior exposure.",
                "caution": "n=9 only; quantitative trend-support signal, but not routine-comparative evidence.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/33155070/",
            },
            {
                "study": "Larkins 2021",
                "year": 2021,
                "pmid": 33988769,
                "family": family_ramps,
                "subgroup": "Minimally invasive RAMPS",
                "design": "Scoping review",
                "quantitative_role": "Review-only heterogeneity source",
                "used_for_probability": "No",
                "used_for_access_map": "No",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "Laparoscopic RAMPS node counts ranged from 11 to 43 across small series.",
                "caution": "Review of small retrospective reports; not a controlled comparison.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/33988769/",
            },
            {
                "study": "Ishida 2022",
                "year": 2022,
                "pmid": 34837132,
                "family": family_std,
                "subgroup": "Tumor-location logic",
                "design": "Anatomical oncologic analysis",
                "quantitative_role": "No direct approach comparison",
                "used_for_probability": "No",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "No",
                "main_message": "Non-peripancreatic nodal metastasis concentrated in more proximal tumors.",
                "caution": "Defines extension logic, not route superiority.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/34837132/",
            },
            {
                "study": "Minagawa 2022",
                "year": 2022,
                "pmid": 35847444,
                "family": family_std,
                "subgroup": "Tumor-location quantitative baseline",
                "design": "Location-stratified retrospective cohort",
                "quantitative_role": "Estimable baseline cohorts",
                "used_for_probability": "Yes",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "No",
                "main_message": "Body 16 [13-22], proximal tail 18 [14-25], distal tail 15 [11-18].",
                "caution": "Important location confounder; do not pool as if body and distal tail are identical.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/35847444/",
            },
            {
                "study": "Tanaka 2022",
                "year": 2022,
                "pmid": 36139569,
                "family": family_std,
                "subgroup": "Tumor-location logic",
                "design": "Pattern-of-metastasis analysis",
                "quantitative_role": "No direct ELN threshold model",
                "used_for_probability": "No",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "No",
                "main_message": "Body and tail tumors should not be treated as sharing the same optimal dissection field.",
                "caution": "Supports location stratification rather than route superiority.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/36139569/",
            },
            {
                "study": "Seo 2022",
                "year": 2022,
                "pmid": 35182251,
                "family": family_std,
                "subgroup": "Distal-tail station logic",
                "design": "Tail-cancer nodal pattern study",
                "quantitative_role": "No direct approach comparison",
                "used_for_probability": "No",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "No",
                "main_message": "Stations 7/8/9 showed 0% metastasis in distal-tail PDAC in this series.",
                "caution": "Best applied to distal-tail disease only.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/35182251/",
            },
            {
                "study": "Kwon 2024",
                "year": 2024,
                "pmid": 38672628,
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "design": "Propensity-matched comparative cohort",
                "quantitative_role": "Estimable matched cohorts",
                "used_for_probability": "Yes",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "After PSM, RAMPS yielded 15 versus 10 nodes for cDPS; 5-year OS remained similar.",
                "caution": "Good comparator design, but still retrospective.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/38672628/",
            },
            {
                "study": "Hirashita 2024",
                "year": 2024,
                "pmid": 38957555,
                "family": family_std,
                "subgroup": "Distal-tail station logic",
                "design": "Distal-tail recurrence analysis",
                "quantitative_role": "No direct approach comparison",
                "used_for_probability": "No",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "No",
                "main_message": "Common-hepatic-artery dissection had limited oncologic value in distal-tail PDAC.",
                "caution": "Do not generalize the distal-tail conclusion to all left-sided tumors.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/38957555/",
            },
            {
                "study": "Liu 2025",
                "year": 2025,
                "pmid": 39996303,
                "family": family_std,
                "subgroup": "Randomized standard-DP benchmark",
                "design": "Randomized trial",
                "quantitative_role": "Estimable standard cohorts",
                "used_for_probability": "Yes",
                "used_for_access_map": "No",
                "used_for_heterogeneity_audit": "No",
                "main_message": "Randomized LDP versus ODP reported median ELN 13.5 [10-20] and 12 [7-17].",
                "caution": "Standard-route benchmark only; does not test RAMPS or deep posterior variants.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/39996303/",
            },
            {
                "study": "Yin 2025",
                "year": 2025,
                "pmid": 40533171,
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "design": "Propensity-matched comparative cohort",
                "quantitative_role": "Estimable RAMPS cohort",
                "used_for_probability": "Yes",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "RAMPS median ELN was 16 [11-21] after matching; OS HR 0.87 and not statistically superior.",
                "caution": "Matched comparator ELN dispersion not fully reported in the abstract.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/40533171/",
            },
            {
                "study": "Zhou 2026",
                "year": 2026,
                "pmid": 41924794,
                "family": family_ramps,
                "subgroup": "Meta-analysis",
                "design": "Meta-analysis",
                "quantitative_role": "Pooled comparative support",
                "used_for_probability": "No",
                "used_for_access_map": "Yes",
                "used_for_heterogeneity_audit": "Yes",
                "main_message": "Sixteen studies: RAMPS improved LN harvest (MD +3.83) and R0, but OS HR remained 0.87.",
                "caution": "Meta-level support for direction of effect, but not a patient-level threshold model.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/41924794/",
            },
        ]
    )

    cohorts = pd.DataFrame(
        [
            {
                "family": family_std,
                "subgroup": "Matched conventional DP",
                "cohort_label": "cDPS | Kwon 2024",
                "study": "Kwon 2024",
                "year": 2024,
                "n": 99,
                "summary_type": "median_range",
                "center": 10.0,
                "low": 0.0,
                "high": 36.0,
                "design": "PSM comparator arm",
                "tumor_context": "Matched left-sided PDAC, mixed body/tail",
                "os_signal": "Neutral versus RAMPS",
            },
            {
                "family": family_std,
                "subgroup": "Randomized standard DP",
                "cohort_label": "ODP | Liu 2025",
                "study": "Liu 2025",
                "year": 2025,
                "n": 129,
                "summary_type": "median_iqr",
                "center": 12.0,
                "low": 7.0,
                "high": 17.0,
                "design": "Randomized arm",
                "tumor_context": "Resectable left-sided pancreatic cancer",
                "os_signal": "Not a RAMPS comparator",
            },
            {
                "family": family_std,
                "subgroup": "Randomized standard DP",
                "cohort_label": "LDP | Liu 2025",
                "study": "Liu 2025",
                "year": 2025,
                "n": 130,
                "summary_type": "median_iqr",
                "center": 13.5,
                "low": 10.0,
                "high": 20.0,
                "design": "Randomized arm",
                "tumor_context": "Resectable left-sided pancreatic cancer",
                "os_signal": "Not a RAMPS comparator",
            },
            {
                "family": family_std,
                "subgroup": "Location-stratified baseline",
                "cohort_label": "Distal tail | Minagawa 2022",
                "study": "Minagawa 2022",
                "year": 2022,
                "n": 24,
                "summary_type": "median_iqr",
                "center": 15.0,
                "low": 11.0,
                "high": 18.0,
                "design": "Location-stratified cohort",
                "tumor_context": "Distal tail PDAC",
                "os_signal": "Location benchmark",
            },
            {
                "family": family_std,
                "subgroup": "Location-stratified baseline",
                "cohort_label": "Body | Minagawa 2022",
                "study": "Minagawa 2022",
                "year": 2022,
                "n": 58,
                "summary_type": "median_iqr",
                "center": 16.0,
                "low": 13.0,
                "high": 22.0,
                "design": "Location-stratified cohort",
                "tumor_context": "Pancreatic body PDAC",
                "os_signal": "Location benchmark",
            },
            {
                "family": family_std,
                "subgroup": "Location-stratified baseline",
                "cohort_label": "Proximal tail | Minagawa 2022",
                "study": "Minagawa 2022",
                "year": 2022,
                "n": 38,
                "summary_type": "median_iqr",
                "center": 18.0,
                "low": 14.0,
                "high": 25.0,
                "design": "Location-stratified cohort",
                "tumor_context": "Proximal-tail PDAC",
                "os_signal": "Location benchmark",
            },
            {
                "family": family_std,
                "subgroup": "Comparative counter-arm",
                "cohort_label": "DPS | Sham 2020",
                "study": "Sham 2020",
                "year": 2020,
                "n": 193,
                "summary_type": "point_only",
                "center": 18.0,
                "low": 18.0,
                "high": 18.0,
                "design": "Dual-institution comparator arm",
                "tumor_context": "Mixed left-sided PDAC",
                "os_signal": "Neutral versus RAMPS",
            },
            {
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "cohort_label": "RAMPS | Sham 2020",
                "study": "Sham 2020",
                "year": 2020,
                "n": 253,
                "summary_type": "point_only",
                "center": 12.0,
                "low": 12.0,
                "high": 12.0,
                "design": "Dual-institution RAMPS cohort",
                "tumor_context": "Mixed left-sided PDAC",
                "os_signal": "Neutral versus standard DP",
            },
            {
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "cohort_label": "RAMPS | Kwon 2024",
                "study": "Kwon 2024",
                "year": 2024,
                "n": 99,
                "summary_type": "median_range",
                "center": 15.0,
                "low": 5.0,
                "high": 51.0,
                "design": "PSM RAMPS cohort",
                "tumor_context": "Matched left-sided PDAC, mixed body/tail",
                "os_signal": "Neutral versus cDPS",
            },
            {
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "cohort_label": "RAMPS | Yin 2025",
                "study": "Yin 2025",
                "year": 2025,
                "n": 174,
                "summary_type": "median_iqr",
                "center": 16.0,
                "low": 11.0,
                "high": 21.0,
                "design": "PSM RAMPS cohort",
                "tumor_context": "Matched left-sided PDAC",
                "os_signal": "OS HR 0.87, not significant",
            },
            {
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "cohort_label": "RAMPS | Kim 2016",
                "study": "Kim 2016",
                "year": 2016,
                "n": 30,
                "summary_type": "mean_sd",
                "center": 21.5,
                "low": 13.2,
                "high": 29.8,
                "design": "Comparative RAMPS cohort",
                "tumor_context": "Single-center left-sided PDAC",
                "os_signal": "Neutral versus conventional DP",
            },
            {
                "family": family_ramps,
                "subgroup": "Open/comparative RAMPS",
                "cohort_label": "RAMPS | Abe 2016",
                "study": "Abe 2016",
                "year": 2016,
                "n": 53,
                "summary_type": "mean_sd",
                "center": 28.4,
                "low": 16.8,
                "high": 40.0,
                "design": "Comparative RAMPS cohort",
                "tumor_context": "Single-center left-sided PDAC",
                "os_signal": "OS trend favorable but not definitive",
            },
            {
                "family": family_ramps,
                "subgroup": "Minimally invasive RAMPS",
                "cohort_label": "Lap-RAMPS | Kim 2017",
                "study": "Kim 2017",
                "year": 2017,
                "n": 15,
                "summary_type": "mean_sd",
                "center": 18.1,
                "low": 11.9,
                "high": 24.3,
                "design": "Single-arm Lap-RAMPS",
                "tumor_context": "Selected minimally invasive RAMPS",
                "os_signal": "No direct comparator",
            },
            {
                "family": family_deep,
                "subgroup": "Left-posterior artery-first",
                "cohort_label": "Left-posterior | Nagai 2020",
                "study": "Nagai 2020",
                "year": 2020,
                "n": 9,
                "summary_type": "median_range",
                "center": 35.0,
                "low": 11.0,
                "high": 49.0,
                "design": "Single-arm posterior-route series",
                "tumor_context": "Highly selected expert-center cases",
                "os_signal": "No direct comparator",
            },
        ]
    )

    cohorts["family_order"] = cohorts["family"].map(family_order)
    cohorts["color"] = cohorts["family"].map(palette)
    cohorts["eligible_for_probability"] = cohorts["summary_type"] != "point_only"
    estimated = cohorts.apply(estimate_mean_sd, axis=1, result_type="expand")
    estimated.columns = ["mean_equiv", "sd_equiv", "harmonization_method"]
    cohorts = pd.concat([cohorts, estimated], axis=1)

    def original_summary(row: pd.Series) -> str:
        if row["summary_type"] == "mean_sd":
            return f"{row['center']:.1f}±{(row['high'] - row['low']) / 2.0:.1f}"
        if row["summary_type"] == "median_iqr":
            return f"{row['center']:.1f} [{row['low']:.1f}-{row['high']:.1f}]"
        if row["summary_type"] == "median_range":
            return f"{row['center']:.1f} ({row['low']:.1f}-{row['high']:.1f})"
        return f"{row['center']:.1f} (point only)"

    cohorts["reported_summary"] = cohorts.apply(original_summary, axis=1)
    cohorts["harmonized_summary"] = cohorts.apply(
        lambda row: "NA" if not row["eligible_for_probability"] else f"{row['mean_equiv']:.1f}±{row['sd_equiv']:.1f}",
        axis=1,
    )

    for thr in config.thresholds:
        cohorts[f"p_ge_{thr}"] = cohorts.apply(
            lambda row: prob_ge_threshold(row["mean_equiv"], row["sd_equiv"], thr) if row["eligible_for_probability"] else np.nan,
            axis=1,
        )
        bounds = cohorts.apply(
            lambda row: ci_from_probability(row[f"p_ge_{thr}"], int(row["n"])) if row["eligible_for_probability"] else (np.nan, np.nan),
            axis=1,
            result_type="expand",
        )
        bounds.columns = [f"p_ge_{thr}_low", f"p_ge_{thr}_high"]
        cohorts = pd.concat([cohorts, bounds], axis=1)
        cohorts[f"estimated_n_ge_{thr}"] = cohorts[f"p_ge_{thr}"] * cohorts["n"]

    family_pool = summarize_family_distributions(cohorts, list(config.thresholds))
    family_meta = pd.DataFrame(
        [
            {
                "family": family_std,
                "family_order": 1,
                "color": palette[family_std],
                "recommendation_class": "Baseline reference",
                "safety_feasibility": "Most reproducible routine option; distal-tail extension beyond routine stations is often unnecessary.",
                "case_mix_caution": "Family pool contains randomized standard arms plus body/proximal-tail and distal-tail subgroups; interpret as an unadjusted benchmark.",
                "clinical_message": "Combined family distribution is a baseline benchmark. It usually supports ELN 12-14, but the 21-node zone remains difficult to reach consistently.",
            },
            {
                "family": family_ramps,
                "family_order": 2,
                "color": palette[family_ramps],
                "recommendation_class": "Preferred escalation",
                "safety_feasibility": "Best-supported standardized escalation when posterior margin logic or higher ELN target is desired; OS superiority remains unproven.",
                "case_mix_caution": "Open/comparative RAMPS and minimally invasive RAMPS are intentionally tracked as subgroups because yield is heterogeneous.",
                "clinical_message": "Combined family distribution shifts rightward relative to standard DP and increases the estimated chance of crossing ELN=21, while remaining an associative rather than causal inference.",
            },
            {
                "family": family_deep,
                "family_order": 3,
                "color": palette[family_deep],
                "recommendation_class": "Trend-supported, low-certainty",
                "safety_feasibility": "Posterior/artery-first techniques show a favorable quantitative yield trend, but current evidence remains center-dependent and low certainty.",
                "case_mix_caution": "Quantitative signal is driven by one small estimable cohort plus technical reports, with substantial surgeon-selection bias.",
                "clinical_message": "Current literature supports a posterior/artery-first technical spectrum with favorable ELN trend signal, but certainty is low and superiority over standard DP or RAMPS is not established.",
            },
        ]
    )
    family_pool = family_pool.merge(family_meta, on="family", how="left")

    support_counts = (
        study_catalog.assign(
            is_quantitative=study_catalog["quantitative_role"].str.contains("Estimable|Point-only|Pooled", regex=True),
            is_comparative=study_catalog["design"].str.contains(r"\bcomparative\b|randomized|propensity|meta-analysis", case=False, regex=True),
            is_meta_review=study_catalog["design"].str.contains(r"review|meta-analysis", case=False, regex=True),
            is_access=study_catalog["used_for_access_map"].eq("Yes"),
        )
        .groupby("family", sort=False)
        .agg(
            total_papers=("study", "nunique"),
            comparative_papers=("is_comparative", "sum"),
            meta_review_papers=("is_meta_review", "sum"),
            access_support_papers=("is_access", "sum"),
        )
        .reset_index()
    )
    family_pool = family_pool.merge(support_counts, on="family", how="left")
    family_pool["evidence_rule_met"] = (family_pool["estimable_cohorts"] >= 3) & (family_pool["estimable_studies"] >= 3)
    family_pool = family_pool.sort_values("family_order").reset_index(drop=True)

    std_p21 = float(family_pool.loc[family_pool["family"] == family_std, "p_ge_21"].iloc[0])
    std_p14 = float(family_pool.loc[family_pool["family"] == family_std, "p_ge_14"].iloc[0])
    std_p12 = float(family_pool.loc[family_pool["family"] == family_std, "p_ge_12"].iloc[0])
    family_pool["rr21_vs_standard"] = family_pool["p_ge_21"] / std_p21
    family_pool["rd21_vs_standard"] = family_pool["p_ge_21"] - std_p21
    family_pool["rr14_vs_standard"] = family_pool["p_ge_14"] / std_p14
    family_pool["rr12_vs_standard"] = family_pool["p_ge_12"] / std_p12
    family_pool["combined_summary"] = family_pool.apply(
        lambda row: format_mean_sd(row["combined_mean"], row["combined_sd"]),
        axis=1,
    )
    family_pool["p_ge_12_text"] = family_pool["p_ge_12"].map(pct_text)
    family_pool["p_ge_14_text"] = family_pool["p_ge_14"].map(pct_text)
    family_pool["p_ge_21_text"] = family_pool.apply(
        lambda row: prob_ci_text(row["p_ge_21"], row["p_ge_21_low"], row["p_ge_21_high"]),
        axis=1,
    )
    family_pool["part1_survival_association"] = family_pool.apply(
        lambda row: (
            f"Part 1 associated ELN>=12 / >=14 / >=21 with 60-mo ΔRMST "
            f"+{anchors['rmst_12_60']:.2f} / +{anchors['rmst_14_60']:.2f} / +{anchors['rmst_21_60']:.2f} months. "
            f"For this family, {anchor_position_text(row['combined_mean'], anchors['change_point'])}; "
            f"distribution-derived P(ELN>=21) is {pct_text(row['p_ge_21'])}."
        ),
        axis=1,
    )

    stat_validation = pairwise_family_tests(family_pool)
    deep_note = pd.DataFrame(
        [
            {
                "comparison": f"{family_deep} vs all formal families",
                "threshold": "Not formally tested",
                "left_combined_mean": np.nan,
                "right_combined_mean": np.nan,
                "left_probability": np.nan,
                "right_probability": np.nan,
                "risk_difference": np.nan,
                "risk_ratio": np.nan,
                "rr_low": np.nan,
                "rr_high": np.nan,
                "z_value": np.nan,
                "approx_p_value": np.nan,
                "note": "Posterior / artery-first spectrum has quantitative trend support but <3 estimable quantitative cohorts and no robust comparative dataset, so it is not entered into formal pairwise testing.",
            }
        ]
    )
    stat_validation = pd.concat([stat_validation, deep_note], ignore_index=True)

    ramps_subgroup_pool = summarize_family_distributions(
        cohorts[cohorts["family"] == family_ramps].copy().assign(family=lambda df: df["subgroup"]),
        list(config.thresholds),
    ).rename(columns={"family": "ramps_subgroup"})
    ramps_subgroup_pool["combined_summary"] = ramps_subgroup_pool.apply(
        lambda row: format_mean_sd(row["combined_mean"], row["combined_sd"]),
        axis=1,
    )

    reference_summary = pd.concat(
        [
            family_pool[
                [
                    "family",
                    "total_papers",
                    "comparative_papers",
                    "meta_review_papers",
                    "quantitative_papers",
                    "estimable_studies",
                    "estimable_cohorts",
                    "combined_n",
                    "combined_summary",
                    "p_ge_21",
                    "p_ge_21_low",
                    "p_ge_21_high",
                    "formal_testable",
                    "evidence_rule_met",
                ]
            ].assign(level="Family"),
            ramps_subgroup_pool.assign(level="RAMPS subgroup").rename(columns={"ramps_subgroup": "family"})[
                [
                    "family",
                    "estimable_studies",
                    "estimable_cohorts",
                    "combined_n",
                    "quantitative_papers",
                    "combined_summary",
                    "formal_testable",
                    "p_ge_21",
                    "p_ge_21_low",
                    "p_ge_21_high",
                    "level",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    station_access = pd.DataFrame(
        [
            {
                "family": family_std,
                "context": "Body/proximal tail standard DP",
                "evidence_count": 3,
                "quantitative_direct_papers": 1,
                "station_10_11_18": "Routine",
                "station_9": "Selective",
                "station_14": "Selective",
                "posterior_plane": "Limited",
                "location_note": "Proximal tumors justify broader fields more often than distal-tail tumors.",
                "supporting_refs": "Minagawa 2022; Tanaka 2022; Ishida 2022",
            },
            {
                "family": family_std,
                "context": "Distal-tail standard DP",
                "evidence_count": 3,
                "quantitative_direct_papers": 1,
                "station_10_11_18": "Routine",
                "station_9": "Usually low-yield",
                "station_14": "Usually low-yield",
                "posterior_plane": "Limited",
                "location_note": "Distal-tail disease often does not justify routine suprapancreatic extension.",
                "supporting_refs": "Minagawa 2022; Seo 2022; Hirashita 2024",
            },
            {
                "family": family_ramps,
                "context": "Comparative/open-or-mixed RAMPS",
                "evidence_count": 5,
                "quantitative_direct_papers": 4,
                "station_10_11_18": "Routine",
                "station_9": "Selective to enhanced",
                "station_14": "Enhanced",
                "posterior_plane": "Enhanced",
                "location_note": "Posterior margin and station-14 clearance are integral to the intended plane, but distal-tail benefit is not uniform.",
                "supporting_refs": "Abe 2016; Kim 2016; Kwon 2024; Yin 2025; Zhou 2026",
            },
            {
                "family": family_deep,
                "context": "Left-posterior / artery-first variants",
                "evidence_count": 3,
                "quantitative_direct_papers": 1,
                "station_10_11_18": "Routine",
                "station_9": "Enhanced",
                "station_14": "Direct",
                "posterior_plane": "Direct",
                "location_note": "Best technical exposure of SMA-side and posterior plane, but supported almost entirely by expert-center reports.",
                "supporting_refs": "Takaori 2016; Ome 2019; Nagai 2020",
            },
        ]
    )

    heterogeneity_audit = pd.DataFrame(
        [
            {"study": "Abe 2016", "route_group": "RAMPS", "subgroup": "Open/comparative", "design_strength": 1, "dispersion_reported": 1, "direct_comparator": 1, "eln_direction": 2, "os_direction": 1, "safety_feasibility": 2},
            {"study": "Kim 2016", "route_group": "RAMPS", "subgroup": "Open/comparative", "design_strength": 1, "dispersion_reported": 1, "direct_comparator": 1, "eln_direction": 2, "os_direction": 1, "safety_feasibility": 2},
            {"study": "Sham 2020", "route_group": "RAMPS", "subgroup": "Open/comparative", "design_strength": 1, "dispersion_reported": 0, "direct_comparator": 1, "eln_direction": 0, "os_direction": 1, "safety_feasibility": 2},
            {"study": "Kwon 2024", "route_group": "RAMPS", "subgroup": "Open/comparative", "design_strength": 2, "dispersion_reported": 1, "direct_comparator": 1, "eln_direction": 2, "os_direction": 1, "safety_feasibility": 2},
            {"study": "Yin 2025", "route_group": "RAMPS", "subgroup": "Open/comparative", "design_strength": 2, "dispersion_reported": 1, "direct_comparator": 1, "eln_direction": 2, "os_direction": 1, "safety_feasibility": 2},
            {"study": "Kim 2017", "route_group": "RAMPS", "subgroup": "Minimally invasive", "design_strength": 0, "dispersion_reported": 1, "direct_comparator": 0, "eln_direction": 1, "os_direction": 1, "safety_feasibility": 1},
            {"study": "Larkins 2021", "route_group": "RAMPS", "subgroup": "Minimally invasive review", "design_strength": 1, "dispersion_reported": 1, "direct_comparator": 0, "eln_direction": 1, "os_direction": 1, "safety_feasibility": 1},
            {"study": "Nagai 2020", "route_group": "Posterior / artery-first spectrum", "subgroup": "Left-posterior", "design_strength": 0, "dispersion_reported": 1, "direct_comparator": 0, "eln_direction": 2, "os_direction": 1, "safety_feasibility": 1},
            {"study": "Zhou 2026", "route_group": "RAMPS", "subgroup": "Meta-analysis", "design_strength": 3, "dispersion_reported": 1, "direct_comparator": 1, "eln_direction": 2, "os_direction": 1, "safety_feasibility": 2},
        ]
    )

    heterogeneity_legend = pd.DataFrame(
        [
            {"field": "design_strength", "coding": "0=single arm/technical; 1=retrospective comparator or review; 2=PSM/randomized comparator; 3=meta-analysis"},
            {"field": "dispersion_reported", "coding": "0=no spread reported for threshold modeling; 1=spread available or inferable"},
            {"field": "direct_comparator", "coding": "0=no direct standard comparator; 1=direct comparator available"},
            {"field": "eln_direction", "coding": "0=ELN favors standard/mixed against advanced route; 1=no directional conclusion; 2=ELN favors advanced route"},
            {"field": "os_direction", "coding": "0=OS favors standard; 1=neutral/not reported; 2=OS favors advanced route"},
            {"field": "safety_feasibility", "coding": "0=problematic/not acceptable; 1=unclear or expert-center only; 2=reported as feasible/comparable"},
        ]
    )

    claim_audit = pd.DataFrame(
        [
            {"issue": "Core figure focus", "prior_problem": "Only ELN centers were shown, so the script could not answer which route helps more patients reach ELN>=21.", "current_fix": "Figure 6 now reports only family-level threshold attainment derived from combined mean/SD, avoiding mixed cohort-level and family-level panels in the same figure."},
            {"issue": "Part 1 linkage", "prior_problem": "ELN values were detached from survival logic or were overextended into new survival arithmetic.", "current_fix": "Table 5 and Figure 7 now only associate family-level threshold attainment with the already established Part 1 RMST deltas; no probability×RMST extrapolation remains."},
            {"issue": "Deep-route overclaim", "prior_problem": "Sparse small-series evidence was treated like stable superiority.", "current_fix": "Posterior / artery-first spectrum is now framed as a trend-supported but low-certainty technical family and remains excluded from formal pairwise testing."},
            {"issue": "Mixed summary statistics", "prior_problem": "Means, medians, and ranges were shown side-by-side without a common scale.", "current_fix": "All threshold probabilities use harmonized mean/SD-equivalent estimates; point-only studies are listed but not pooled into family distributions."},
            {"issue": "RAMPS heterogeneity", "prior_problem": "RAMPS was treated as a single undifferentiated block.", "current_fix": "Open/comparative and minimally invasive RAMPS are tracked as separate subgroups in the reference summary and heterogeneity audit."},
            {"issue": "Family pooling method", "prior_problem": "Family-level results were previously based on weighted averages of study-specific probabilities.", "current_fix": "Family-level results now first combine mean and SD within each family and only then derive threshold attainment from the combined distribution."},
        ]
    )

    method_notes = pd.DataFrame(
        [
            {"Item": "Population frame", "Detail": "Part 2 remains a literature-derived, summary-data synthesis; it is not a patient-level meta-analysis."},
            {"Item": "Family combination rule", "Detail": "Family-level synthesis first combines harmonized mean and SD across cohorts using sample-size-weighted pooled moments, then derives P(ELN>=12/14/21) from the combined distribution."},
            {"Item": "Probability model", "Detail": "Threshold attainment is derived from the combined family distribution using a Gaussian approximation with continuity correction."},
            {"Item": "Inclusion for pooling", "Detail": "Only cohorts with a center and spread estimate enter family distribution combination. Point-only papers remain qualitative counterexamples."},
            {"Item": "Formal comparison rule", "Detail": "Family-level formal tests require at least 3 estimable cohorts. Posterior / artery-first spectrum therefore remains trend-supported but low-certainty and is not formally compared."},
            {"Item": "Part 1 linkage", "Detail": "Part 1 RMST deltas are cited only as established anchor associations. No new survival benefit is calculated from family-level probabilities."},
            {"Item": "Confounding", "Detail": "Tumor location, surgeon experience, and case selection are not fully adjustable from published summaries; route effects are associations, not causal estimates."},
            {"Item": "Time window", "Detail": "Current synthesis uses 2016-2026 PubMed-indexed literature, matching the manuscript scope used in this revision."},
        ]
    )

    recommended_refs = study_catalog[["study", "year", "pmid", "family", "subgroup", "url"]].copy()

    table3 = cohorts.sort_values(["family_order", "eligible_for_probability", "year", "study", "cohort_label"], ascending=[True, False, True, True, True]).copy()
    table3["P(ELN>=12)"] = table3.apply(lambda row: prob_ci_text(row["p_ge_12"], row["p_ge_12_low"], row["p_ge_12_high"]), axis=1)
    table3["P(ELN>=14)"] = table3.apply(lambda row: prob_ci_text(row["p_ge_14"], row["p_ge_14_low"], row["p_ge_14_high"]), axis=1)
    table3["P(ELN>=21)"] = table3.apply(lambda row: prob_ci_text(row["p_ge_21"], row["p_ge_21_low"], row["p_ge_21_high"]), axis=1)
    table3["Eligibility note"] = table3.apply(
        lambda row: "Estimable"
        if row["eligible_for_probability"]
        else "Point-only summary: shown for heterogeneity, excluded from family distribution combination",
        axis=1,
    )
    table3_main = table3[
        [
            "family",
            "cohort_label",
            "n",
            "design",
            "reported_summary",
            "harmonized_summary",
            "P(ELN>=12)",
            "P(ELN>=14)",
            "P(ELN>=21)",
        ]
    ].rename(columns={
        "family": "Approach family",
        "cohort_label": "Cohort",
        "design": "Design",
        "reported_summary": "Reported ELN",
        "harmonized_summary": "Harmonized mean±SD",
    })

    table4_main = station_access[
        ["family", "context", "station_10_11_18", "station_9", "station_14", "posterior_plane", "supporting_refs"]
    ].rename(
        columns={
            "family": "Approach family",
            "context": "Context",
            "station_10_11_18": "#10/#11/#18",
            "station_9": "#9",
            "station_14": "#14",
            "posterior_plane": "Posterior plane",
            "supporting_refs": "Key references",
        }
    )

    table5_main = family_pool[
        [
            "family",
            "estimable_cohorts",
            "combined_n",
            "combined_summary",
            "p_ge_12_text",
            "p_ge_14_text",
            "p_ge_21_text",
            "rr21_vs_standard",
            "recommendation_class",
            "clinical_message",
        ]
    ].rename(
        columns={
            "family": "Approach family",
            "estimable_cohorts": "Estimable cohorts",
            "combined_n": "Combined n",
            "combined_summary": "Combined mean±SD",
            "p_ge_12_text": "P(ELN≥12)",
            "p_ge_14_text": "P(ELN≥14)",
            "p_ge_21_text": "P(ELN≥21)",
            "rr21_vs_standard": "RR vs Standard (ELN≥21)",
            "recommendation_class": "Recommendation",
            "clinical_message": "Interpretation",
        }
    )
    table5_main["RR vs Standard (ELN≥21)"] = table5_main.apply(
        lambda row: "Not formally estimated"
        if row["Approach family"] == family_deep
        else ("1.00" if np.isclose(row["RR vs Standard (ELN≥21)"], 1.0, equal_nan=False) else (f"{row['RR vs Standard (ELN≥21)']:.2f}" if np.isfinite(row["RR vs Standard (ELN≥21)"]) else "NA")),
        axis=1,
    )

    return {
        "claim_audit": claim_audit,
        "study_catalog": study_catalog,
        "reference_summary": reference_summary,
        "usage_map": study_catalog.rename(columns={"study": "文献", "family": "所属家族", "quantitative_role": "证据定位", "main_message": "核心可引用信息", "caution": "使用警戒"}),
        "fig4_source": cohorts,
        "fig5_access": station_access,
        "fig6_scores": heterogeneity_audit,
        "fig6_rationale": heterogeneity_legend,
        "fig10_ramps": table3.sort_values(["family_order", "year", "study"]),
        "fig10_source_basis": family_pool,
        "fig12_quant": family_pool,
        "fig12_qual": family_meta,
        "table3_approaches": table3_main,
        "table4_access": table4_main,
        "table5_threshold": table5_main,
        "table6_deep": claim_audit,
        "method_notes": method_notes,
        "stat_validation": stat_validation,
        "recommended_refs": recommended_refs,
    }


def plot_figure4(cohorts: pd.DataFrame, anchors: dict[str, float], config: Config) -> None:
    data = cohorts.sort_values(["family_order", "mean_equiv", "center"], ascending=[True, True, True]).reset_index(drop=True)
    n_rows = len(data)
    fig, ax = plt.subplots(figsize=(10, 0.52 * n_rows + 1.6))
    y = np.arange(n_rows)[::-1]

    prev_family = None
    for yi, (_, row) in zip(y, data.iterrows()):
        if prev_family is not None and row["family"] != prev_family:
            ax.axhline(yi + 0.5, color="#CCCCCC", linewidth=0.7, linestyle="-")
        prev_family = row["family"]
        if row["eligible_for_probability"]:
            ax.errorbar(
                row["mean_equiv"], yi, xerr=row["sd_equiv"],
                fmt="o", color=row["color"], ecolor=row["color"],
                elinewidth=1.5, capsize=3, markersize=6, zorder=3,
            )
        else:
            ax.scatter(row["center"], yi, marker="D", s=35, color=row["color"], alpha=0.65, zorder=3)
        label_x = (row["mean_equiv"] + row["sd_equiv"] + 1.0) if row["eligible_for_probability"] else (row["center"] + 1.0)
        label_x = min(label_x, 44)
        ax.text(label_x, yi, row["reported_summary"], va="center", ha="left", fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([fill(label, 28) for label in data["cohort_label"]], fontsize=8)
    ax.set_xlabel("Harmonized ELN (mean ± SD equivalent)", fontsize=10)
    ax.set_title("Published cohort ELN landscape by approach family", fontsize=12, pad=12)

    anchor_specs = [
        ("best_cut", "#C0392B", "ELN = 12"),
        ("hr1_display", "#D4760A", "ELN ≈ 14"),
        ("change_point", "#2471A3", "ELN = 21"),
    ]
    top_transform = blended_transform_factory(ax.transData, ax.transAxes)
    for name, color, label_text in anchor_specs:
        xpos = anchors[name]
        ax.axvline(xpos, color=color, linestyle="--", linewidth=1.2, alpha=0.8)

    offsets = [0.97, 0.93, 0.97]
    for (name, color, label_text), yoff in zip(anchor_specs, offsets):
        ax.text(anchors[name], yoff, label_text, color=color, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", transform=top_transform)

    ax.set_xlim(3, 52)
    ax.set_xticks(np.arange(5, 51, 5))
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    estimable_handle = plt.Line2D([], [], color="#555555", marker="o", linestyle="None", markersize=5.5, label="Estimable cohort (mean ± SD)")
    point_handle = plt.Line2D([], [], color="#555555", marker="D", linestyle="None", markersize=4.5, label="Point-only (no spread)")
    ax.legend(
        handles=[estimable_handle, point_handle],
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.68),
        fontsize=7.5,
        handletextpad=0.4,
    )

    fig.tight_layout()
    save_figure(fig, config.outdir / "figures" / PART2_MAIN_FIGURES[0], config.dpi)


def plot_figure5(station_access: pd.DataFrame, config: Config) -> None:
    access_scale = {
        "Usually low-yield": 0, "Limited": 0,
        "Selective": 1, "Selective to enhanced": 2,
        "Routine": 2, "Enhanced": 3, "Direct": 4,
    }
    columns = ["station_10_11_18", "station_9", "station_14", "posterior_plane"]
    labels = ["#10/#11/#18", "#9", "#14", "Posterior plane"]
    family_order = {family: idx for idx, family in enumerate(station_access["family"].drop_duplicates(), start=1)}

    def summarize_group(sub: pd.DataFrame, label_builder) -> dict[str, object]:
        row: dict[str, object] = {
            "family": sub["family"].iloc[0],
            "family_order": family_order[sub["family"].iloc[0]],
            "row_label": label_builder(sub),
        }
        for col in columns:
            scores = sub[col].map(access_scale)
            max_score = float(scores.max())
            first_idx = scores[scores == max_score].index[0]
            row[f"{col}_score"] = max_score
            row[col] = sub.loc[first_idx, col]
        return row

    family_rows = []
    for family, sub in station_access.groupby("family", sort=False):
        family_rows.append(summarize_group(sub, lambda grp: family_display_name(grp["family"].iloc[0])))
    family_df = pd.DataFrame(family_rows).sort_values("family_order").reset_index(drop=True)

    family_matrix = family_df[[f"{col}_score" for col in columns]].to_numpy(dtype=float)
    family_annotations = family_df[columns].to_numpy(dtype=str)
    family_labels_list = family_df["row_label"].tolist()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    cmap = ListedColormap(["#E8F0F7", "#B8D4E8", "#FDE8C8", "#F5C882", "#D4820A"])
    im = ax.imshow(family_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(np.arange(len(family_labels_list)))
    ax.set_yticklabels(family_labels_list, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Anatomical station accessibility by approach family", fontsize=11, pad=10)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.6)

    for i in range(family_matrix.shape[0]):
        for j in range(family_matrix.shape[1]):
            txt_color = "white" if family_matrix[i, j] >= 3 else "#333333"
            ax.text(j, i, family_annotations[i, j], ha="center", va="center", fontsize=8, color=txt_color)

    ax.set_xticks(np.arange(-0.5, family_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, family_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.06, shrink=0.9)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["Low-yield", "Selective", "Routine", "Enhanced", "Direct"])
    cbar.ax.tick_params(labelsize=7.5)

    fig.tight_layout()
    save_figure(fig, config.outdir / "figures" / PART2_MAIN_FIGURES[1], config.dpi)


def plot_figure6(cohorts: pd.DataFrame, family_pool: pd.DataFrame, config: Config) -> None:
    families = family_pool.sort_values("family_order").reset_index(drop=True)
    n_families = len(families)
    x = np.arange(n_families)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5))

    colors_12, colors_14, colors_21 = "#6BAED6", "#FD8D3C", "#D62728"
    bars12 = ax.bar(x - width, families["p_ge_12"] * 100, width=width, color=colors_12,
                    edgecolor="white", linewidth=0.5, label="P(ELN ≥ 12)")
    bars14 = ax.bar(x, families["p_ge_14"] * 100, width=width, color=colors_14,
                    edgecolor="white", linewidth=0.5, label="P(ELN ≥ 14)")
    bars21 = ax.bar(x + width, families["p_ge_21"] * 100, width=width, color=colors_21,
                    edgecolor="white", linewidth=0.5, label="P(ELN ≥ 21)")

    deep_mask = families["family"].eq("Posterior / artery-first spectrum").to_numpy()
    for bars in [bars12, bars14, bars21]:
        for idx, bar in enumerate(bars):
            if deep_mask[idx]:
                bar.set_hatch("///")
                bar.set_alpha(0.45)
                bar.set_edgecolor("#999999")

    for xpos, (_, row) in zip(x, families.iterrows()):
        val = row["p_ge_21"] * 100
        ax.text(xpos + width, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    xticklabels = [family_display_name(name) for name in families["family"]]
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel("Estimated probability (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Family-level threshold attainment probability", fontsize=12, pad=12)

    descriptive_handle = patches.Patch(facecolor="#CCCCCC", edgecolor="#999999", hatch="///", label="Low-certainty (trend only)")
    legend_handles = [
        patches.Patch(facecolor=colors_12, label="P(ELN ≥ 12)"),
        patches.Patch(facecolor=colors_14, label="P(ELN ≥ 14)"),
        patches.Patch(facecolor=colors_21, label="P(ELN ≥ 21)"),
        descriptive_handle,
    ]
    ax.legend(
        handles=legend_handles,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.tight_layout(rect=[0, 0.16, 1, 1])
    save_figure(fig, config.outdir / "figures" / PART2_MAIN_FIGURES[2], config.dpi)


def plot_figure7(family_pool: pd.DataFrame, anchors: dict[str, float], config: Config) -> None:
    data = family_pool.sort_values("family_order").reset_index(drop=True)
    n_families = len(data)
    fig, ax = plt.subplots(figsize=(8, 3.6))

    y = np.arange(n_families)[::-1]

    anchor_specs = [
        ("best_cut", "#C0392B", "ELN = 12"),
        ("hr1_display", "#D4760A", "ELN ≈ 14"),
        ("change_point", "#2471A3", "ELN = 21"),
    ]
    for name, color, _ in anchor_specs:
        ax.axvline(anchors[name], color=color, linestyle="--", linewidth=1.1, alpha=0.7)

    for yi, (_, row) in zip(y, data.iterrows()):
        marker_style = dict(fmt="o", ecolor=row["color"], color=row["color"],
                            capsize=3.5, elinewidth=1.8, markersize=7, zorder=3)
        if row["family"] == "Posterior / artery-first spectrum":
            marker_style["fmt"] = "s"
            marker_style["markersize"] = 6
        ax.errorbar(row["combined_mean"], yi, xerr=row["combined_sd"], **marker_style)

        p21_text = row["p_ge_21_text"].replace("-", "–")
        label_x = min(row["combined_mean"] + row["combined_sd"] + 0.8, 45)
        ax.text(label_x, yi, f"P(≥21): {p21_text}", va="center", ha="left", fontsize=7.5, color="#333333")

    top_transform = blended_transform_factory(ax.transData, ax.transAxes)
    offsets = [0.97, 0.93, 0.97]
    for (name, color, label_text), yoff in zip(anchor_specs, offsets):
        ax.text(anchors[name], yoff, label_text, color=color, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", transform=top_transform)

    ax.set_yticks(y)
    ylabels = [family_display_name(name) for name in data["family"]]
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Combined family ELN distribution (mean ± SD)", fontsize=10)
    ax.set_title("Family-level ELN distributions versus Part 1 survival anchors", fontsize=11, pad=10)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.5, n_families - 0.5)
    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save_figure(fig, config.outdir / "figures" / PART2_MAIN_FIGURES[3], config.dpi)


def plot_supplementary_figure(heterogeneity_audit: pd.DataFrame, config: Config) -> None:
    display = heterogeneity_audit.copy()
    score_cols = ["design_strength", "dispersion_reported", "direct_comparator", "eln_direction", "os_direction", "safety_feasibility"]
    matrix = display[score_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = ListedColormap(["#F0F0F0", "#B8D4E8", "#6BAED6", "#2171B5"])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    col_labels = ["Design\nstrength", "Spread\nreported", "Direct\ncomparator", "ELN\ndirection", "OS\ndirection", "Safety/\nfeasibility"]
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(np.arange(len(display)))
    subgroup_abbrev = {
        "Open/comparative": "Open", "Minimally invasive": "MI",
        "Minimally invasive review": "MI review", "Left-posterior": "Posterior",
        "Meta-analysis": "Meta",
    }
    ax.set_yticklabels([f"{row.study} ({subgroup_abbrev.get(row.subgroup, row.subgroup)})" for row in display.itertuples()], fontsize=8.5)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.5)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt_color = "white" if val >= 2.5 else "#333333"
            ax.text(j, i, int(val) if np.isfinite(val) else "", ha="center", va="center", fontsize=9, color=txt_color)
    ax.set_title("Heterogeneity audit of surgical route evidence", fontsize=11, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(["0", "1", "2", "3"])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Score", fontsize=9)
    fig.tight_layout()
    save_figure(fig, config.outdir / "figures" / PART2_SUPPLEMENTARY_FIGURES[0], config.dpi)


def plot_supplementary_anchor_matrix(family_pool: pd.DataFrame, anchors: dict[str, float], config: Config) -> None:
    data = family_pool.sort_values("family_order").reset_index(drop=True)
    matrix = data[["p_ge_12", "p_ge_14", "p_ge_21"]].to_numpy(dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ylabels = [family_display_name(name) for name in data["family"]]
    threshold_labels = [
        f"ELN ≥ 12\n(ΔRMST +{anchors['rmst_12_60']:.2f} mo)",
        f"ELN ≥ 14\n(ΔRMST +{anchors['rmst_14_60']:.2f} mo)",
        f"ELN ≥ 21\n(ΔRMST +{anchors['rmst_21_60']:.2f} mo)",
    ]
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(threshold_labels, fontsize=8)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Anchor-attainment matrix linked to Part 1 ΔRMST", fontsize=11, pad=10)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            txt_color = "white" if value >= 55 else "#222222"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color=txt_color)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    for idx, family in enumerate(data["family"]):
        if family == "Posterior / artery-first spectrum":
            ax.add_patch(patches.Rectangle((-0.5, idx - 0.5), 3, 1, fill=False, edgecolor="#888888", linewidth=1.0, linestyle="--"))

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.05, shrink=0.9)
    cbar.set_label("Probability (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    save_figure(fig, config.outdir / "figures" / PART2_SUPPLEMENTARY_FIGURES[1], config.dpi)


def write_workbook(data: dict[str, pd.DataFrame], config: Config) -> None:
    workbook_path = config.outdir / "tables" / "Part2_Anatomical_Evidence_Tables.xlsx"
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = list(PART2_SHEET_NAME_MAP.keys())
    remaining = [key for key in data.keys() if key not in ordered_keys]
    with pd.ExcelWriter(workbook_path) as writer:
        for key in ordered_keys + remaining:
            if key not in data:
                continue
            df = data[key]
            sheet_name = PART2_SHEET_NAME_MAP.get(key, key)
            df.to_excel(writer, sheet_name=sanitize_sheet_name(sheet_name), index=False)


def write_text_outputs(data: dict[str, pd.DataFrame], anchors: dict[str, float], config: Config) -> None:
    family_pool = data["fig12_quant"]
    lines = [
        "Part 2 Output Index",
        "",
        "Main figures:",
        f"- {PART2_MAIN_FIGURES[0]}: published cohort ELN landscape with Part 1 anchor lines.",
        f"- {PART2_MAIN_FIGURES[1]}: family overview plus representative PubMed references for anatomical access.",
        f"- {PART2_MAIN_FIGURES[2]}: family-level threshold probabilities after first combining mean/SD within each family.",
        f"- {PART2_MAIN_FIGURES[3]}: family distributions positioned against Part 1 anchor thresholds; posterior spectrum remains low-certainty and is not formally pairwise tested.",
        "",
        "Supplementary figures:",
        f"- {PART2_SUPPLEMENTARY_FIGURES[0]}: objective heterogeneity audit of route evidence.",
        f"- {PART2_SUPPLEMENTARY_FIGURES[1]}: anchor-attainment matrix linked to Part 1 ΔRMST.",
        "",
        "Main tables:",
        "- Table3_Approach_Comparison: harmonized cohort summaries and threshold probabilities.",
        "- Table4_Anatomical_Access: station access, evidence counts, and location cautions.",
        "- Table5_Anchor_Attainment: combined family mean/SD, threshold probabilities, RR vs standard, and Part 1 anchor association.",
        "",
        "Key anchors imported from Part 1:",
        f"- ELN=12 delta RMST at 60 months: {anchors['rmst_12_60']:.2f} months",
        f"- ELN≈{int(round(anchors['hr1_display']))} delta RMST at 60 months: {anchors['rmst_14_60']:.2f} months",
        f"- ELN=21 delta RMST at 60 months: {anchors['rmst_21_60']:.2f} months",
        "",
        "Current family-level ELN>=21 estimates from combined family distributions:",
    ]
    for row in family_pool.itertuples():
        lines.append(f"- {row.family}: mean±SD={row.combined_summary}, P(ELN>=21)={row.p_ge_21_text}, recommendation={row.recommendation_class}")
    (config.outdir / "Output_Index.txt").write_text("\n".join(lines), encoding="utf-8")

    notes = [
        "Part 2 methodological notes",
        "",
        "1. Family-level synthesis first combines harmonized mean and SD across cohorts and only then derives threshold attainment.",
        "2. Threshold probabilities remain summary-data approximations, not patient-level reconstructions.",
        "3. Point-only cohorts remain descriptive counterexamples and do not enter family distribution combination.",
        "4. Posterior / artery-first spectrum shows quantitative trend support, but it is not formally compared because the estimable evidence base remains too sparse.",
        "5. Figure 7 caption should retain that posterior spectrum is low-certainty and not formally pairwise tested, even though that statement is no longer printed inside the plotting area.",
        "6. Figure 5 main text version now shows family overview plus representative references; the full evidence structure remains in tables/workbook.",
        "7. RAMPS is split into open/comparative and minimally invasive subgroups in the supporting workbook to expose heterogeneity.",
    ]
    (config.outdir / "Part2_Anatomical_Evidence_Notes.txt").write_text("\n".join(notes), encoding="utf-8")


def cleanup_obsolete_outputs(config: Config) -> None:
    obsolete = [
        config.outdir / "Part2_Rewrite_Guide.md",
    ]
    for path in obsolete:
        if path.exists():
            path.unlink()


def main() -> None:
    config = Config()
    apply_style()
    (config.outdir / "figures").mkdir(parents=True, exist_ok=True)
    (config.outdir / "tables").mkdir(parents=True, exist_ok=True)
    anchors = load_part1_anchors(config)
    data = build_data(config, anchors)
    plot_figure4(data["fig4_source"], anchors, config)
    plot_figure5(data["fig5_access"], config)
    plot_figure6(data["fig4_source"], data["fig12_quant"], config)
    plot_figure7(data["fig12_quant"], anchors, config)
    plot_supplementary_figure(data["fig6_scores"], config)
    plot_supplementary_anchor_matrix(data["fig12_quant"], anchors, config)
    write_workbook(data, config)
    write_text_outputs(data, anchors, config)
    cleanup_obsolete_outputs(config)
    print("Part 2 anatomical evidence outputs regenerated.")


if __name__ == "__main__":
    main()
