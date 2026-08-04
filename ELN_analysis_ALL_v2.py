# -*- coding: utf-8 -*-
"""
# =============================================================================
# Project: ELN Analysis Pipeline for Pancreatic Cancer (SEER Cohort)
# Purpose:
#     This script performs a comprehensive evaluation of the prognostic value of Examined Lymph Nodes (ELN) in pancreatic cancer patients. It integrates
#     survival analysis, non-linear modeling, ELN binning, threshold optimization,
#     subgroup analysis, and visualization to identify the optimal ELN range and
#     investigate its association with overall survival.
# Main Functions:
#   1. Data Processing
#       - Load SEER-derived dataset
#       - Clean missing values and recode clinical variables
#       - Generate ELN bin groups (custom size or step=1)
#   2. Descriptive Analysis
#       - ELN distribution (histogram + KDE)
#       - ELN summary statistics
#   3. Survival Analysis
#       - Kaplan–Meier (KM) curves for ELN bins
#       - Median OS comparison across ELN groups
#       - smoothed median OS curve using step=1 ELN values
#   4. Restricted Cubic Spline (RCS) Modeling
#       - Fit Cox model with ELN as a non-linear predictor
#       - Visualize HR vs. ELN using spline-based curve
#   5. Subgroup Analysis
#       - KM curves by Summary Stage (Localized/Regional/Distant/Unknown)
#       - KM curves by SurgeryExtent (partial, subtotal, total pancreatectomy)
#       - Median OS comparison within each subgroup
#   6. ELN Threshold Search
#       - Loop through candidate cutoffs (10–40 or custom range)
#       - Generate KM curves for each possible ELN cut
#       - Evaluate survival curve separation
# Outputs:
#       - Histograms, density plots
#       - KM plots for overall and subgroup analyses
#       - Median OS comparison (bar + line charts)
#       - RCS curve of ELN vs. hazard ratio
#       - Threshold KM plots for each ELN cutoff
#       - Study flowchart figure
# Notes:
#       - This script is designed for reproducible ELN survival modeling.
# =============================================================================
"""
import os, re, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

CONFIG = {
    "excel_path": "Pancreatic_Body_Tail_ALL.xlsx",
    "sheet_name": "export",
    "outdir": "outputs_ALL_ELN_v2",

    # —— 运行模式（快/全）——
    "fast_mode": False,
    "do_rcs": True,

    # 自助/网格（快/全挡）
    "bootstrap_n_cindex_fast": 60,
    "bootstrap_n_cindex_full": 200,
    "bootstrap_n_cp_fast": 60,
    "bootstrap_n_cp_full": 200,
    "cp_grid_fast": list(range(24, 37)),
    "cp_grid_full": list(range(15, 45)),
    "dca_points_fast": 40,
    "dca_points_full": 80,
    "strategy_methods_fast": ["OW"],
    "strategy_methods_full": ["OW", "IPTW"],
    "strategy_cuts_fast": [10, 30],
    "strategy_cuts_full": [10, 20, 30, 40],

    # 其他
    "random_state": 42,
    "rcs_df": 4,
    "rcs_knots": None,
    "eln_search_range": list(range(6, 36)),
    "rmst_times": [12, 36, 60],
    "calib_times": [12, 36, 60],
    "plot_dpi": 200,
    "font_family_zh": "Microsoft YaHei",

    # 纳入标准：仍剔除分期 Other；手术范围保留小类 other（约10例）
    "drop_stage_other": True,
    "drop_surgery_other": False,

    # 倾向评分协变量（存在才会使用）
    "ps_covars": ["Age", "Sex_bin", "Stage", "Primary_part", "SurgeryExtent", "chemo_bin", "radiation_bin"],
}

# ============ 依赖 ============
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index, restricted_mean_survival_time
from lifelines.exceptions import ConvergenceError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
try:
    from patsy import dmatrix
    HAS_PATSY = True
except Exception:
    HAS_PATSY = False
    print("⚠️ If patsy is not installed, RCS will be skipped (you may install it with pip install patsy).")

# ============ 工具 ============
def ensure_dir(p):
    p = Path(p); p.mkdir(parents=True, exist_ok=True)

def writer_open(path_excel):
    return pd.ExcelWriter(Path(path_excel), engine="xlsxwriter")

def save_fig(path):
    plt.tight_layout()
    plt.savefig(Path(path), dpi=CONFIG["plot_dpi"])
    plt.close()

# ========= 通用小工具（供多个补丁调用） =========
def _to_num_series(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", "").str.strip(), errors="coerce")

def save_fig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = [CONFIG["font_family_zh"]]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

def to_num(s): return pd.to_numeric(s, errors="coerce")

# ==== KM 汇总：n、Events、KM 中位、S(12/36/60) ====
def km_summary_table(sub, time_col, event_col, time_points=(12,36,60)):
    sub = sub.dropna(subset=[time_col, event_col])
    n = len(sub)
    if n == 0:
        row = {"n":0, "Events":0, "Median(m)":np.nan}
        for t in time_points: row[f"S({t}m)"]=np.nan
        return row
    kmf = KaplanMeierFitter()
    kmf.fit(sub[time_col].values, sub[event_col].values)
    med = kmf.median_survival_time_
    row = {"n":int(n), "Events":int(sub[event_col].sum()), "Median(m)": float(med) if med==med else np.nan}
    sf = kmf.survival_function_
    tt = sf.index.to_numpy()
    for t in time_points:
        t_closest = tt[np.argmin(np.abs(tt - t))]
        row[f"S({t}m)"] = float(sf.loc[t_closest].values[0])
    return row

def plot_km_groups(df_sub, time_col, event_col, group_col, title, png_name, order=None):
    if order is None: order = sorted(df_sub[group_col].dropna().unique().tolist())
    set_chinese_font()
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7.2,5.6))
    for g in order:
        sub = df_sub[df_sub[group_col]==g].dropna(subset=[time_col, event_col])
        if len(sub) < 5: continue
        kmf.fit(sub[time_col], sub[event_col], label=f"{g} (n={len(sub)})")
        kmf.plot(ci_show=False)
    plt.title(title); plt.xlabel("Survival months"); plt.ylabel("Survival probability")
    plt.grid(True, alpha=0.3)
    save_fig(Path(CONFIG["outdir"])/png_name)

# ============ 读取与派生 ============
def load_and_prepare():
    print("✅ Begin reading data ...")
    fp = Path(CONFIG["excel_path"])
    if not fp.exists():
        raise FileNotFoundError(f"No data files found：{fp.resolve()}")
    xls = pd.ExcelFile(fp)
    sheet = CONFIG["sheet_name"] if CONFIG["sheet_name"] in xls.sheet_names else xls.sheet_names[0]
    df_raw = pd.read_excel(fp, sheet_name=sheet)
    df = df_raw.copy()
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    # 关键列模糊匹配
    candidates = {
        "ELN": ["regional nodes examined"],
        "POS": ["regional nodes positive"],
        "SurvM": ["survival months"],
        "Vital": ["vital status recode"],
        "Stage": ["summary stage 2000"],
        "SurgPrim": ["rx summ--surg prim site"],
        "SurgOther": ["surg oth reg/dis", "rx summ--surg oth reg/dis"],
        "Age": ["age"],
        "Primary": ["primary site - labeled", "primary site"],
        "Chemo": ["chemotherapy recode"],
        "Radiation": ["radiation recode"],
        "Sex": ["sex"]
    }
    found = {}
    for k, pats in candidates.items():
        for col in df.columns:
            cl = col.lower()
            if any(p in cl for p in pats):
                found[k] = col; break

    # 数值化
    for key in ["ELN","POS","SurvM","Age"]:
        if key in found: df[key] = to_num(df[found[key]])
    # ===== PATCH-1: 清洗 ELN/POS & 合法性 & 计算 LNR =====
    ELN_COL = "ELN"; POS_COL = "POS"
    bad_codes = {90, 95, 97, 98, 99, 126}
    for col in [ELN_COL, POS_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col].isin(bad_codes), col] = np.nan
            df.loc[df[col] < 0, col] = np.nan

    # 删除 POS>ELN 的异常行（两列都合法时）
    before = len(df)
    mask_bad = df[POS_COL].notna() & df[ELN_COL].notna() & (df[POS_COL] > df[ELN_COL])
    df = df.loc[~mask_bad].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"[Cleaning] Deleted anomalous rows from POS>ELN：{removed}")

    # 统一计算 LNR（仅当 ELN>0 & POS>=0）
    df["LNR"] = np.where(
        (df.get(ELN_COL, np.nan) > 0) & (df.get(POS_COL, np.nan) >= 0),
        df.get(POS_COL) / df.get(ELN_COL), np.nan
    )

    # Event
    if "Vital" in found:
        v = df[found["Vital"]].astype(str).str.lower().str.strip()
        num_v = pd.to_numeric(v, errors="coerce")
        event = np.where(v.str.contains("dead"), 1,
                 np.where(v.str.contains("alive"), 0,
                 np.where(num_v.notna(), (num_v>0).astype(int), np.nan)))
        df["Event"] = to_num(event)
    else:
        df["Event"] = np.nan

    # Stage（保持原有 4 类；不造 Other）
    if "Stage" in found:
        df["Stage"] = df[found["Stage"]].astype(str).str.strip()

    # Primary 部位
    def map_primary(x):
        s = str(x).lower()
        if "tail" in s: return "Tail"
        if "body" in s: return "Body"
        if "nos" in s:  return "NOS"
        return np.nan
    if "Primary" in found:
        df["Primary_part"] = df[found["Primary"]].map(map_primary)

    # Sex 二元
    if "Sex" in found:
        s = df[found["Sex"]].astype(str).str.lower().str.strip()
        df["Sex_bin"] = np.where(s.str.contains("male") & ~s.str.contains("female"), 1, 0)

    # Chemo 二元
    if "Chemo" in found:
        c = df[found["Chemo"]].astype(str).str.lower().str.strip()
        df["chemo_bin"] = np.where(c.str.contains("yes"), 1, np.where(c.str.contains("no"), 0, np.nan))

    # Radiation 二元（原始值为 "Beam radiation"/"None/Unknown" 等文本）
    if "Radiation" in found:
        r = df[found["Radiation"]].astype(str).str.lower().str.strip()
        df["radiation_bin"] = np.where(
            r.str.contains("beam|radio|radia|brachytherapy|isotope", regex=True) & ~r.str.contains("none|refused|unknown|recommended", regex=True),
            1,
            np.where(r.str.contains("none|refused|unknown|recommended", regex=True), 0, np.nan)
        )

    # LNR
    if "ELN" in df and "POS" in df:
        df["LNR"] = np.where(df["ELN"]>0, df["POS"]/df["ELN"], np.nan)

    # 手术范围：优先使用你表中现成列（如 “SurgeryExtent” 或 “手术范围”）
    surgery_col = None
    for name in df.columns:
        low = name.lower()
        if low == "surgeryextent" or ("Surgery" in name and "Scope" in name):
            surgery_col = name; break
    if surgery_col is not None:
        df["SurgeryExtent"] = df[surgery_col].astype(str).str.strip()
    else:
        # 没有现成列时，保底映射（不会覆盖你的 AX 列）
        def parse_code_num(x):
            m = re.search(r"(\d+)", str(x)); return int(m.group(1)) if m else np.nan
        df["SurgPrim_code"] = df[found["SurgPrim"]].map(parse_code_num) if "SurgPrim" in found else np.nan
        if "SurgOther" in found:
            so = df[found["SurgOther"]].astype(str).str.extract(r"(\d+)")[0]
            df["SurgOther_code"] = pd.to_numeric(so, errors="coerce")
        else:
            df["SurgOther_code"] = np.nan
        def map_surgery_row(row):
            sp = row.get("SurgPrim_code", np.nan)
            so = row.get("SurgOther_code", np.nan)
            if pd.notna(so) and so>0: return "Combined"
            if pd.isna(sp): return "other"
            if 30 <= sp <= 40: return "Local"
            if sp >= 60: return "Extended"
            return "other"
        df["SurgeryExtent"] = df.apply(map_surgery_row, axis=1)

    # 手术范围「其他」统一为小写 other（与稿件一致）；Combined/Local/Extended 不变
    if "SurgeryExtent" in df.columns:
        se = df["SurgeryExtent"].astype(str).str.strip()
        df["SurgeryExtent"] = se.replace({"Other": "other", "OTHER": "other"})

    # —— 可用样本：可选剔除 Stage Other；手术 other 默认保留 —— #
    base = df["SurvM"].notna() & df["Event"].notna() & (df["SurvM"]>0)
    usable = base
    if CONFIG.get("drop_stage_other", True):
        usable = usable & (~df["Stage"].astype(str).str.strip().eq("Other"))
    if CONFIG.get("drop_surgery_other", True):
        usable = usable & (~df["SurgeryExtent"].astype(str).str.strip().str.lower().eq("other"))

    print(f"📊 Original number of lines={df.shape[0]} | Sample size for survival analysis={usable.sum()}")
    return df, found, usable

# ===== PATCH-4: F1 ELN 分布图 =====
def plot_eln_hist_density(df, eln_col="ELN", out_path=None):
    s = df[eln_col]
    # 关键容错：若同名列导致 DataFrame，取第一列（或可改成多数非缺失的那列）
    if isinstance(s, pd.DataFrame):
        # 也可用：s = s.iloc[:, s.notna().sum().values.argmax()] 选择非缺失更多的那列
        s = s.iloc[:, 0]
    x = _to_num_series(s)
    x = x[np.isfinite(x)]
    if x.empty:
        print("[F1] ELN is empty, skip"); return
    set_chinese_font()
    plt.figure(figsize=(7,4.5))
    ax = plt.gca()
    x.plot(kind="hist", bins=40, density=True, alpha=0.45, edgecolor="black", ax=ax)
    try:
        x.plot(kind="kde", linewidth=2, ax=ax)
    except Exception as e:
        print("[F1] KDE Failure：", e)
    ax.set_title("ELN Distribution (Histogram + Kernel Density)")
    ax.set_xlabel("ELN"); ax.set_ylabel("density"); ax.grid(alpha=0.3)
    save_fig(out_path or Path(CONFIG["outdir"])/"F1_ELN_hist_density.png")

# ============ ELN 分箱（最新版口径）& 步长=1 ============
def eln_binning_and_plots(df, usable, outdir):
    print("📦 Generate ELN binning/stratification (latest version) and step size = 1 km median OS curve ...")
    ensure_dir(outdir)
    df_surv = df.loc[usable].copy()
    # 先画 ELN 分布图
    plot_eln_hist_density(df_surv, eln_col="ELN", out_path=Path(outdir)/"F1_ELN_hist_density.png")

    # 固定分箱（与最新版一致）
    bins   = [0, 5, 10, 15, 20, 30, 40, 60, 9999]
    labels = ["1-5", "6-10", "11-15", "16-20", "21-30", "31-40", "41-60", "61+"]  # :contentReference[oaicite:3]{index=3}
    df_surv["ELN_bin"] = pd.cut(df_surv["ELN"], bins=bins, labels=labels)

    # [1] 总体分箱表 + KM + 中位OS图
    print("\n[1] Generate overall ELN partitioning table...")
    rows = []
    for lb in labels:
        sub = df_surv[df_surv["ELN_bin"]==lb]
        info = km_summary_table(sub, "SurvM", "Event", (12,36,60))
        info["ELN sub-box"] = lb
        rows.append(info)
    overall_tbl = pd.DataFrame(rows)[["ELN sub-box","n","Events","Median(m)","S(12m)","S(36m)","S(60m)"]]
    overall_tbl.to_excel(Path(outdir)/"ELN_bins_overall.xlsx", index=False)

    plot_km_groups(df_surv, "SurvM", "Event", "ELN_bin",
                   "KM：ELN Subdivision (Overall)",
                   "KM_ELN_bins_overall.png",
                   order=labels)  # :contentReference[oaicite:4]{index=4}

    set_chinese_font()
    plt.figure(figsize=(7.2,5.0))
    plt.bar(overall_tbl["ELN sub-box"], overall_tbl["Median(m)"])
    plt.title("ELN Subgrouping and Median Survival (Overall)")
    plt.xlabel("ELN sub-box"); plt.ylabel("Median OS (months)")
    save_fig(Path(outdir)/"Barchart_Median_by_ELNbin_overall.png")

    plt.figure(figsize=(7.2,5.0))
    plt.plot(overall_tbl["ELN sub-box"], overall_tbl["Median(m)"], marker="o")
    plt.title("ELN Subgrouping and Median Survival (Overall, Line Chart)")
    plt.xlabel("ELN sub-box"); plt.ylabel("Median OS (months)")
    save_fig(Path(outdir)/"Line_Median_by_ELNbin_overall.png")

    # [2] 按分期分层（不含 Other，本脚本也不会产生 Other；Unknown 可保留或后续选择不展示）
    print("\n[2] Generate ELN partitioning tables stratified by Summary stage 2000 and KM.")
    stages_order = ["Localized", "Regional", "Distant", "Unknown"]
    with writer_open(Path(outdir)/"ELN_bins_by_stage.xlsx") as w:
        for st in stages_order:
            sub_stage = df_surv[df_surv["Stage"].astype(str).str.contains(st, case=False, regex=False)]
            if len(sub_stage)==0: continue
            rows = []
            for lb in labels:
                sub = sub_stage[sub_stage["ELN_bin"]==lb]
                info = km_summary_table(sub, "SurvM", "Event", (12,36,60))
                info["ELN sub-box"] = lb
                rows.append(info)
            tdf = pd.DataFrame(rows)[["ELN sub-box","n","Events","Median(m)","S(12m)","S(36m)","S(60m)"]]
            tdf.to_excel(w, sheet_name=(f"Stage_{st}"[:31]), index=False)
            plot_km_groups(sub_stage, "SurvM", "Event", "ELN_bin",
                           f"KM：ELN sub-box（{st} stratification）",
                           f"KM_ELN_bins_Stage_{st}.png",
                           order=labels)  # :contentReference[oaicite:5]{index=5}

            # 条形图
            set_chinese_font()
            plt.figure(figsize=(7.2,5.0))
            plt.bar(tdf["ELN sub-box"], tdf["Median(m)"])
            plt.title(f"ELN Subgrouping and Median Survival（{st} stratification）"); plt.xlabel("ELN sub-box"); plt.ylabel("Median OS (months)")
            save_fig(Path(outdir)/f"Barchart_Median_by_ELNbin_Stage_{st}.png")

    # [3] 按手术范围分层（含 other 组，n≥5 才出 KM 曲线）
    print("\n[3] Generate ELN stratification tables and KM by surgical scope.")
    with writer_open(Path(outdir)/"ELN_bins_by_surgery.xlsx") as w:
        for sg in sorted(df_surv["SurgeryExtent"].dropna().unique(), key=lambda x: str(x).lower()):
            sub_sg = df_surv[df_surv["SurgeryExtent"]==sg]
            if len(sub_sg)==0: continue
            rows = []
            for lb in labels:
                sub = sub_sg[sub_sg["ELN_bin"]==lb]
                info = km_summary_table(sub, "SurvM", "Event", (12,36,60))
                info["ELN sub-box"] = lb
                rows.append(info)
            tdf = pd.DataFrame(rows)[["ELN sub-box","n","Events","Median(m)","S(12m)","S(36m)","S(60m)"]]
            tdf.to_excel(w, sheet_name=(f"Surgery_{sg}"[:31]), index=False)
            plot_km_groups(sub_sg, "SurvM", "Event", "ELN_bin",
                           f"KM：ELN sub-box（Surgical group：{sg}）",
                           f"KM_ELN_bins_Surgery_{sg}.png",
                           order=labels)  # :contentReference[oaicite:6]{index=6}

            set_chinese_font()
            plt.figure(figsize=(7.2,5.0))
            plt.bar(tdf["ELN sub-box"], tdf["Median(m)"])
            plt.title(f"ELN Subgrouping and Median Survival（Surgical group：{sg}）"); plt.xlabel("ELN sub-box"); plt.ylabel("Median OS (months)")
            save_fig(Path(outdir)/f"Barchart_Median_by_ELNbin_Surgery_{sg}.png")

    # [4] 步长=1：逐点 KM 中位 + 7 点滚动中位
    step_rows = []
    for e in sorted(df_surv["ELN"].dropna().astype(int).unique()):
        sub = df_surv[df_surv["ELN"]==e]
        row = km_summary_table(sub, "SurvM", "Event", ())
        step_rows.append({"ELN": e, "Median_OS": row["Median(m)"]})
    step_df = pd.DataFrame(step_rows).sort_values("ELN")
    step_df["Median_OS_smooth_7pt"] = step_df["Median_OS"].rolling(window=7, center=True, min_periods=1).median()

    set_chinese_font()
    plt.figure(figsize=(6.8,4.5))
    plt.plot(step_df["ELN"], step_df["Median_OS"], lw=1.5, label="KM median OS (raw)")
    plt.plot(step_df["ELN"], step_df["Median_OS_smooth_7pt"], lw=2.2, label="7-point rolling median (smoothed)")
    plt.axvspan(16, 35, color="#cccccc", alpha=0.3, label="Optimal Experience Range (16–35)")
    plt.xlabel("ELN (number of items)"); plt.ylabel("Median overall survival (months)"); plt.title("ELN（step=1）and Median OS")
    plt.legend()
    save_fig(Path(outdir)/"F4_ELN_step1_medianOS_KM.png")

    # 汇总一本 Excel
    used = set()
    with writer_open(Path(outdir)/"ELN_bins_and_strata.xlsx") as w:
        sh = "Overall_ELN_bins"
        overall_tbl.to_excel(w, sheet_name=sh, index=False)
        stage_file = Path(outdir)/"ELN_bins_by_stage.xlsx"
        if stage_file.exists():
            xls = pd.ExcelFile(stage_file)
            for sh0 in xls.sheet_names:
                df_sh = pd.read_excel(stage_file, sheet_name=sh0)
                df_sh.to_excel(w, sheet_name=f"Stage_{sh0}"[:31], index=False)
        surg_file = Path(outdir)/"ELN_bins_by_surgery.xlsx"
        if surg_file.exists():
            xls = pd.ExcelFile(surg_file)
            for sh0 in xls.sheet_names:
                df_sh = pd.read_excel(surg_file, sheet_name=sh0)
                df_sh.to_excel(w, sheet_name=f"Surg_{sh0}"[:31], index=False)

    return overall_tbl, step_df

# ============ 阈值搜索 + KM（保留） ============
def eln_threshold_search_and_km(df, usable, outdir):
    print("🔎 Perform ELN threshold grid search + KM curve ...")
    ensure_dir(outdir)
    rows = []
    for c in CONFIG["eln_search_range"]:
        g1 = (df["ELN"] >= c) & usable
        g0 = (df["ELN"] < c) & usable
        if g1.sum() < 30 or g0.sum() < 30: continue
        res = logrank_test(df.loc[g1,"SurvM"], df.loc[g0,"SurvM"],
                           df.loc[g1,"Event"], df.loc[g0,"Event"])
        rows.append({"threshold": c, "logrank_stat": res.test_statistic, "p_value": res.p_value,
                     "n_high": int(g1.sum()), "n_low": int(g0.sum())})
    grid_df = pd.DataFrame(rows).sort_values("logrank_stat", ascending=False)
    best_cut = int(grid_df.iloc[0]["threshold"]) if not grid_df.empty else 10
    grid_df.head(10).to_excel(Path(outdir)/"T2a_threshold_search_top10.xlsx", index=False)

    def km_plot(cut, tag):
        set_chinese_font()
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(6,4))
        for gval, label in [(0, f"<{cut}"), (1, f"≥{cut}")]:
            mask = usable & ( (df["ELN"]>=cut) if gval==1 else (df["ELN"]<cut) )
            kmf.fit(df.loc[mask,"SurvM"], df.loc[mask,"Event"], label=f"{label} (n={mask.sum()})")
            kmf.plot(ci_show=True)
        plt.xlabel("Follow-up period (months)"); plt.ylabel("Overall survival rate"); plt.title(f"ELN boundary {cut} KM curve")
        save_fig(Path(outdir)/f"F2_KM_ELN_cut{cut}_{tag}.png")

    km_plot(10, "v1"); km_plot(best_cut, "best")
    grid_df.to_excel(Path(outdir)/"T2b_threshold_search_all.xlsx", index=False)
    return best_cut, grid_df

# ============ 设计矩阵构建 & 清洗（不再合并小频到 Other） ============
def build_design_matrix(df, usable):
    base_covars = ["Age", "Sex_bin", "Stage", "Primary_part", "SurgeryExtent", "chemo_bin", "radiation_bin", "ELN"]
    covars = [c for c in base_covars if c in df.columns]

    # 数值列：中位数填补
    numeric_like = [c for c in covars if df[c].dtype.kind in "biufc"]
    for c in numeric_like:
        med = pd.to_numeric(df.loc[usable, c], errors="coerce").median()
        df.loc[usable, c] = pd.to_numeric(df.loc[usable, c], errors="coerce").fillna(med)

    # 分类列：仅处理空值 → "Unknown"；不再合并为 "Other"
    cats_present = [c for c in ["Stage","Primary_part","SurgeryExtent"] if c in covars]
    for cat in cats_present:
        ser = df.loc[usable, cat].astype(str)
        ser = ser.where(~ser.isin(["", "nan", "None"]), other="Unknown")
        df.loc[usable, cat] = ser

    X = pd.get_dummies(df.loc[usable, covars], drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # —— 关键补丁：森林图不显示 Unstaged/Unknown —— #
    drop_cols = [c for c in X.columns
                 if c.startswith("Stage_") and (
                         "unknown" in c.lower() or "unstaged" in c.lower() or "未分期" in c or "未知" in c
                 )]
    X = X.drop(columns=drop_cols, errors="ignore")

    # 连续变量标准化
    for c in ["Age", "ELN"]:
        if c in X.columns:
            mu, sd = X[c].mean(), X[c].std()
            if not np.isfinite(sd) or sd == 0: sd = 1.0
            X[c] = (X[c] - mu) / sd
    y = df.loc[usable, ["SurvM","Event"]].rename(columns={"SurvM":"T","Event":"E"})
    return X, y

def clean_design_matrix_after_dummies(X, outdir, continuous_cols=None):
    if continuous_cols is None:
        continuous_cols = set()
    else:
        continuous_cols = set(continuous_cols)
    n = len(X)
    dropped = {"constant": [], "rare": [], "collinear": []}
    const_cols = [c for c in X.columns if c not in continuous_cols and np.nanstd(X[c].values) == 0]
    if const_cols:
        X = X.drop(columns=const_cols); dropped["constant"].extend(const_cols)
    rare_cols = []
    for c in X.columns:
        if c in continuous_cols:
            continue
        s = X[c].values; ssum = np.nansum(s)
        if (ssum <= 3) or (n - ssum <= 3): rare_cols.append(c)
    if rare_cols:
        rare_cols = list(set(rare_cols)); X = X.drop(columns=rare_cols); dropped["rare"].extend(rare_cols)
    if X.shape[1] > 2:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.999)]
        if to_drop:
            X = X.drop(columns=to_drop); dropped["collinear"].extend(to_drop)
    try:
        with open(Path(outdir)/"T3_dropped_columns.json", "w", encoding="utf-8") as f:
            json.dump(dropped, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return X

# ============ Cox 主模型 + PH + C-index ============
def cox_fit_and_plots(df, usable, outdir):
    print("🧪 Fit Cox proportional hazards model + forest plot + PH test + C-index ...")
    ensure_dir(outdir)
    X, y = build_design_matrix(df, usable)
    X = clean_design_matrix_after_dummies(X, outdir, continuous_cols=["Age", "ELN"])
    data = pd.concat([y, X], axis=1)

    tried, cph = [], None
    for pen in [0.1, 0.5, 1.0, 3.0]:
        try:
            cph_try = CoxPHFitter(penalizer=pen)
            cph_try.fit(data, duration_col="T", event_col="E", robust=True)
            cph = cph_try; tried.append({"penalizer": pen, "ok": True}); break
        except ConvergenceError as e:
            tried.append({"penalizer": pen, "ok": False, "err": str(e)[:180]}); continue

    if cph is None:
        n = len(data)
        keep = []
        for c in X.columns:
            ssum = X[c].sum()
            if (ssum <= 1) or (n - ssum <= 1): continue
            keep.append(c)
        X2 = X[keep].copy(); data2 = pd.concat([y, X2], axis=1)
        for pen in [1.0, 3.0, 5.0]:
            try:
                cph_try = CoxPHFitter(penalizer=pen)
                cph_try.fit(data2, duration_col="T", event_col="E", robust=True)
                cph = cph_try; X = X2; data = data2
                tried.append({"penalizer": pen, "ok": True, "note": "extreme-clean"}); break
            except ConvergenceError as e:
                tried.append({"penalizer": pen, "ok": False, "err": str(e)[:180], "note": "extreme-clean"}); continue

    with open(Path(outdir)/"T3_fit_attempts.json", "w", encoding="utf-8") as f:
        json.dump(tried, f, ensure_ascii=False, indent=2)
    if cph is None:
        raise RuntimeError("The Cox model fitting has yet to converge. The T3_fit_attempts.json file has been generated for troubleshooting purposes.。")

    cph.summary.to_excel(Path(outdir)/"T3_Cox_summary.xlsx")

    set_chinese_font()
    summ = cph.summary.sort_values("exp(coef)")
    plt.figure(figsize=(6, 0.3*len(summ)+2))
    yloc = np.arange(len(summ))
    hr = summ["exp(coef)"].values
    lo = np.exp(summ["coef lower 95%"]); hi = np.exp(summ["coef upper 95%"])
    plt.errorbar(hr, yloc, xerr=[hr-lo, hi-hr], fmt='o', capsize=3)
    plt.axvline(1.0, ls='--', color='grey'); plt.yticks(yloc, summ.index)
    plt.xlabel("Risk ratio（HR）"); plt.title("Cox Proportional Hazards Model Forest Plot")
    save_fig(Path(outdir)/"F6_ForestPlot_Cox.png")
    # —— 论文常用文件名副本 —— #
    try:
        import shutil
        src = Path(outdir) / "F6_ForestPlot_Cox.png"
        dst = Path(outdir) / "ForestPlot_Cox_paperstyle.png"
        if src.exists():
            shutil.copyfile(src, dst)
    except Exception as e:
        print("⚠️ Failed to save paper under common name：", e)

    try:
        ph = proportional_hazard_test(cph, data, time_transform='rank')
        ph.summary.to_excel(Path(outdir)/"T4_PH_test.xlsx")
    except Exception as e:
        print("⚠️ pH test failed：", e)

    rng = np.random.RandomState(CONFIG["random_state"])
    ci_list = []
    for b in range(CONFIG["_BOOT_CINDEX"]):
        idx = rng.choice(data.index, size=len(data), replace=True)
        cph_b = CoxPHFitter(penalizer=getattr(cph, 'penalizer_', 0.0))
        try:
            cph_b.fit(data.loc[idx], duration_col="T", event_col="E", robust=True)
            pred = cph_b.predict_partial_hazard(data.loc[idx])
            c_index = concordance_index(data.loc[idx,"T"], -pred.values, data.loc[idx,"E"])
            ci_list.append(c_index)
        except Exception:
            continue
    pd.Series(ci_list, name="Bootstrap_Cindex").to_excel(Path(outdir)/"T4a_Bootstrap_Cindex.xlsx", index=False)
    # ---- 保存原始 ELN，供校准/DCA里的“规则≥k”使用（避免 data 里找不到 ELN 而报错）----
    try:
        data["_ELN_source"] = pd.to_numeric(df.loc[usable, "ELN"], errors="coerce").values
    except Exception:
        # 即使失败也不影响主流程
        pass
    return cph, data

# ============ RCS ============
def rcs_plot_ELN(df, usable, outdir, cph_data):
    if not HAS_PATSY: return
    print("📈 RCS Nonlinearity: HR-ELN Curve ...")
    ensure_dir(outdir)
    dd = df.loc[usable, ["SurvM","Event","ELN"]].dropna()
    if dd.shape[0] < 100: return

    if CONFIG["rcs_knots"] is None:
        basis = dmatrix(f"bs(ELN, df={CONFIG['rcs_df']}, degree=3, include_intercept=False)",
                        {"ELN": dd["ELN"]}, return_type='dataframe')
    else:
        qs = CONFIG["rcs_knots"]
        basis = dmatrix(f"bs(ELN, knots={qs}, degree=3, include_intercept=False)",
                        {"ELN": dd["ELN"]}, return_type='dataframe')
    d_rcs = pd.concat([dd[["SurvM","Event"]], basis], axis=1)
    cph = CoxPHFitter(); cph.fit(d_rcs, duration_col="SurvM", event_col="Event", robust=True)

    eln_grid = np.linspace(1, np.nanmax(dd["ELN"]), 120)
    b_grid = dmatrix(basis.design_info, {"ELN": eln_grid}, return_type='dataframe')
    ref = np.median(dd["ELN"])
    b_ref = dmatrix(basis.design_info, {"ELN": [ref]}, return_type='dataframe')
    lp_grid = cph.predict_partial_hazard(b_grid).values.flatten()
    lp_ref  = cph.predict_partial_hazard(b_ref).values.flatten()[0]
    hr_rel = lp_grid / lp_ref

    set_chinese_font()
    plt.figure(figsize=(6,4))
    plt.plot(eln_grid, hr_rel, lw=2); plt.axhline(1.0, ls="--", color="grey")
    plt.xlabel("ELN (number of items)"); plt.ylabel("Relative risk（HR）")
    plt.title("RCS: Nonlinear Relationship Between ELN and Risk (Relative to Median)")
    save_fig(Path(outdir)/"F5_RCS_HR_ELN.png")

# ============ 变化点搜索 ============
def changepoint_search(df, usable, outdir, candidates=range(6,36), n_boot=60, penalizer=0.05, seed=42):
    import numpy as np, pandas as pd
    from lifelines import CoxPHFitter
    import json, matplotlib.pyplot as plt
    from pathlib import Path

    # 1) 取主分析集
    if isinstance(usable, pd.DataFrame):
        use = usable.copy()
    else:
        use = df.loc[usable].copy()  # usable 是布尔掩码时

    # 2) 强制数值化
    use["ELN"]   = pd.to_numeric(use.get("ELN"), errors="coerce")
    use["SurvM"] = pd.to_numeric(use.get("SurvM"), errors="coerce")
    use["Event"] = pd.to_numeric(use.get("Event"), errors="coerce")
    use = use.dropna(subset=["SurvM","Event","ELN"])
    use = use[(use["SurvM"] >= 0) & (use["ELN"] >= 0)]

    if len(use) < 100:
        print("⚠️ Insufficient samples available for change point search; skipping.")
        return np.nan, (np.nan, np.nan), pd.DataFrame()

    def get_loglik(cph):
        # lifelines 版本兼容
        val = getattr(cph, "log_likelihood_", None)
        if val is None:
            val = getattr(cph, "_log_likelihood", None)
        return None if val is None else float(val)

    # 3) 网格搜索
    results = []
    for c in candidates:
        tmp = use.copy()
        # ←← 关键：ELN 已数值化，上面已保证
        tmp["ELN_hinge"] = np.maximum(0.0, tmp["ELN"] - float(c))

        data = tmp[["SurvM","Event","ELN","ELN_hinge"]].dropna()
        # 无上段变异（hinge 全0）则跳过
        if (data["ELN_hinge"] > 0).sum() < 5:
            continue

        X = data[["ELN","ELN_hinge"]]
        Y = data[["SurvM","Event"]]
        fit_df = pd.concat([Y, X], axis=1)

        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(fit_df, duration_col="SurvM", event_col="Event", robust=True)
            ll = get_loglik(cph)
            if ll is not None and np.isfinite(ll):
                results.append({"cut": int(c), "loglik": ll})
        except Exception:
            # 拟合失败的切点直接跳过
            continue

    if not results:
        print("⚠️ No valid results found for change point search.")
        return np.nan, (np.nan, np.nan), pd.DataFrame()

    df_grid = pd.DataFrame(results).sort_values("cut")
    best_row = df_grid.sort_values("loglik", ascending=False).iloc[0]
    c_hat = int(best_row["cut"])

    # 4) 自助法CI
    rng = np.random.default_rng(seed)
    idx = np.arange(len(use))
    boots = []
    for b in range(int(n_boot)):
        samp = rng.choice(idx, size=len(idx), replace=True)
        tmpb = use.iloc[samp].copy()
        best_b, best_ll = None, -1e18
        for c in candidates:
            tmpb["ELN_hinge"] = np.maximum(0.0, tmpb["ELN"] - float(c))
            datab = tmpb[["SurvM","Event","ELN","ELN_hinge"]].dropna()
            if (datab["ELN_hinge"] > 0).sum() < 5:
                continue
            cph = CoxPHFitter(penalizer=penalizer)
            try:
                cph.fit(datab, duration_col="SurvM", event_col="Event", robust=True)
                ll = get_loglik(cph)
                if ll is not None and ll > best_ll:
                    best_ll = ll; best_b = int(c)
            except Exception:
                continue
        if best_b is not None:
            boots.append(best_b)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ci = (float(lo), float(hi))
    else:
        ci = (np.nan, np.nan)

    # 5) 导出
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(outdir/"T6_changepoint_loglik_grid.xlsx", engine="xlsxwriter") as w:
        df_grid.to_excel(w, index=False, sheet_name="grid")
    with open(outdir/"T6_changepoint_estimate.json", "w", encoding="utf-8") as f:
        json.dump({"best": c_hat, "boot_CI": list(ci), "n_boot": int(n_boot)}, f, ensure_ascii=False, indent=2)

    # 6) 画“相对最佳”的对数似然差
    delta = df_grid["loglik"] - df_grid["loglik"].max()
    plt.figure(figsize=(6,4), dpi=150)
    if len(df_grid) >= 3:
        plt.plot(df_grid["cut"], delta, marker="o")
        plt.ylabel("Log-likelihood difference (relative to the best)")
    else:
        plt.axvline(c_hat, linestyle="--", color="r", label=f"Change point≈{c_hat}")
        plt.text(c_hat, 0, f"only {len(df_grid)} valid candidate point\nIt is recommended to expand the candidate space or increase the penaliser.", ha="center", va="bottom")
        plt.ylabel("Log-likelihood difference")
    plt.axvline(c_hat, linestyle="--", color="r", label=f"Change point≈{c_hat}")
    plt.xlabel("Candidate change points c"); plt.title("ELN Change Point Search (Segmented Cox)"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir/"F5b_Changepoint_search.png"); plt.close()
    print(f"📦 Change point search completed：best={c_hat}, CI={ci} ；F5b/T6 file exported。")

    return c_hat, ci, df_grid

# ===== PATCH-3: 平衡表（SMD）工具 =====
def _to_dummies_if_categorical(s: pd.Series) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(s): return s.to_frame()
    return pd.get_dummies(s.astype("category"), prefix=s.name, dummy_na=False)

def _weighted_mean_var(x, w):
    w = np.asarray(w, dtype=float); x = np.asarray(x, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0); x = np.where(np.isfinite(x), x, np.nan)
    m = np.isfinite(x); w, x = w[m], x[m]
    if w.sum()==0: return np.nan, np.nan
    mu = np.average(x, weights=w)
    var = np.average((x-mu)**2, weights=w) * (w.sum()/(w.sum()-1)) if w.sum()>1 else np.nan
    return mu, var

def _smd_numeric(x_t, x_c, w_t=None, w_c=None):
    if w_t is None:
        mt, vt = np.nanmean(x_t), np.nanvar(x_t, ddof=1)
    else:
        mt, vt = _weighted_mean_var(x_t, w_t)
    if w_c is None:
        mc, vc = np.nanmean(x_c), np.nanvar(x_c, ddof=1)
    else:
        mc, vc = _weighted_mean_var(x_c, w_c)
    pooled = np.sqrt((vt+vc)/2.0) if (vt>=0 and vc>=0) else np.nan
    return (mt-mc)/pooled if pooled and np.isfinite(pooled) and pooled>0 else np.nan

def compute_smd_table(df_ps, covars, treat_col="TREAT", weight_col=None, truncate_w=None):
    d = df_ps.copy()
    if weight_col is not None and truncate_w is not None:
        lo, hi = truncate_w; d["_w_trunc"] = d[weight_col].clip(lower=lo, upper=hi)
    else:
        d["_w_trunc"] = d[weight_col] if weight_col is not None else None

    rows = []
    for v in covars:
        if v not in d.columns: continue
        X = _to_dummies_if_categorical(d[v])
        for sub in X.columns:
            x_t = X.loc[d[treat_col]==1, sub].values
            x_c = X.loc[d[treat_col]==0, sub].values
            smd_before = _smd_numeric(x_t, x_c, None, None)
            smd_after  = np.nan
            if weight_col is not None:
                w_t = d.loc[d[treat_col]==1, weight_col].values
                w_c = d.loc[d[treat_col]==0, weight_col].values
                smd_after = _smd_numeric(x_t, x_c, w_t, w_c)
            smd_after_tr = np.nan
            if d["_w_trunc"] is not None:
                w_t_tr = d.loc[d[treat_col]==1, "_w_trunc"].values
                w_c_tr = d.loc[d[treat_col]==0, "_w_trunc"].values
                smd_after_tr = _smd_numeric(x_t, x_c, w_t_tr, w_c_tr)
            rows.append({"Variable": sub, "SMD_before": smd_before, "SMD_after": smd_after, "SMD_after_trunc": smd_after_tr})
    res = pd.DataFrame(rows)
    for c in ["SMD_before","SMD_after","SMD_after_trunc"]:
        if c in res.columns: res[c+"_ok(<0.1)"] = res[c].abs() < 0.1
    return res

def export_balance_table(df_ps, covars, method, cut, outdir):
    w = df_ps["W"].replace([np.inf,-np.inf], np.nan).dropna()
    if len(w) >= 50:
        lo, hi = np.nanpercentile(w, [1,99]); hi = min(hi, 10.0)
    else:
        lo, hi = 0.0, 10.0
    tb = compute_smd_table(df_ps, covars, treat_col="TREAT", weight_col="W", truncate_w=(lo,hi))
    fn = Path(outdir)/f"T5_balance_{method}_cut{cut}.xlsx"
    tb.to_excel(fn, index=False); print(f"[Balance sheet export] {fn}")

# ============ 倾向评分 / 策略化 / DCA（同前） ============
def ps_model_and_weights(df, usable, cut, method="OW"):
    d = df.loc[usable].copy()
    d["TREAT"] = (d["ELN"] >= cut).astype(int)
    covs = [c for c in CONFIG["ps_covars"] if c in d.columns]
    X = pd.get_dummies(d[covs], drop_first=True)
    y = d["TREAT"].values
    ps = LogisticRegression(max_iter=1000).fit(X.fillna(0), y).predict_proba(X.fillna(0))[:,1]
    d["PS"] = ps
    if method.upper()=="IPTW":
        p_t = d["TREAT"].mean()
        w = np.where(d["TREAT"]==1, p_t/ps, (1-p_t)/(1-ps))
    else:
        w = np.where(d["TREAT"]==1, 1-ps, ps)
    d["W"] = w
    return d

def weighted_km_and_rmst(d, outdir, cut):
    set_chinese_font()
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for tval, label in [(0, f"ELN < {cut}"), (1, f"ELN ≥ {cut}")]:
        m = (d["TREAT"]==tval)
        kmf.fit(d.loc[m,"SurvM"], d.loc[m,"Event"], weights=d.loc[m,"W"], label=f"{label} (n={m.sum()})")
        kmf.plot(ci_show=False)
    plt.xlabel("Follow-up period (months)"); plt.ylabel("Overall survival rate"); plt.title(f"Weighted KM: Strategy ELN ≥ {cut}")
    save_fig(Path(outdir)/f"F7_KM_weighted_cut{cut}.png")

    rows = []
    for tstar in CONFIG["rmst_times"]:
        rmst_vals = {}
        for tval in [0,1]:
            m = (d["TREAT"]==tval)
            kmf = KaplanMeierFitter()
            kmf.fit(d.loc[m,"SurvM"], d.loc[m,"Event"], weights=d.loc[m,"W"])
            rmst_vals[tval] = restricted_mean_survival_time(kmf, t=tstar)
        rows.append({"cut": cut, "time": tstar,
                     "RMST_treat": rmst_vals[1], "RMST_ctrl": rmst_vals[0],
                     "Delta_RMST": rmst_vals[1]-rmst_vals[0]})
    return pd.DataFrame(rows)

def strategies_block(df, usable, outdir):
    print("🧭 strategic（≥10/20/30/40） + IPTW/OW + 加权KM + ΔRMST ...")
    ensure_dir(outdir)
    all_rows = []
    cuts = CONFIG["_STRAT_CUTS"]; methods = CONFIG["_STRAT_METHS"]
    for cut in cuts:
        for method in methods:
            d = ps_model_and_weights(df, usable, cut, method=method)
            res = weighted_km_and_rmst(d, outdir, cut)
            res["method"] = method
            all_rows.append(res)
            # 导出平衡表（SMD）
            export_balance_table(d, CONFIG["ps_covars"], method, cut, outdir)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_excel(Path(outdir)/"T5_strategy_DeltaRMST.xlsx", index=False)
    return all_df

def binary_at_time_labels_TE(T, E, tau):
    T = pd.to_numeric(T, errors="coerce"); E = pd.to_numeric(E, errors="coerce")
    return np.where((E==1) & (T<=tau), 1, np.where(T>tau, 0, np.nan))

def risk_from_cox(cph, X, t):
    try:
        sf = cph.baseline_survival_.copy()
        tt = sf.index.to_numpy()
        t_closest = tt[np.argmin(np.abs(tt - t))]
        S0_t = sf.loc[t_closest].values.item()  # 修复弃用告警
        lp = cph.predict_partial_hazard(X).values.flatten()
        S_tx = S0_t ** lp
        return 1.0 - S_tx
    except Exception:
        sc = cph.predict_partial_hazard(X).values.flatten()
        return (sc - sc.min())/(sc.max()-sc.min()+1e-9)

def calibration_and_dca(cph, data, outdir):
    print("🎯 Calibration and DCA (12/36/60 months) ...")
    ensure_dir(outdir)
    # 只选用模型真正训练过的列，避免 data 里新增/丢失列导致 predict_partial_hazard 报错
    model_cols = [c for c in cph.params_.index if c in data.columns]
    X = data.loc[:, model_cols]

    for tau in CONFIG["calib_times"]:
        y_series = pd.Series(binary_at_time_labels_TE(data["T"].values, data["E"].values, tau), index=data.index)
        m = ~y_series.isna()
        yb = y_series[m].astype(int); Xb = X.loc[m]
        r = risk_from_cox(cph, Xb, t=tau)

        dfc = pd.DataFrame({"risk": r, "y": yb})
        dfc["decile"] = pd.qcut(dfc["risk"], 10, duplicates="drop")
        tab = dfc.groupby("decile", observed=False).agg(pred=("risk","mean"), obs=("y","mean"))
        tab.to_excel(Path(outdir)/f"T7_calibration_{tau}m.xlsx")

        set_chinese_font()
        plt.figure(figsize=(5,5))
        xs, ys = tab["pred"].values, tab["obs"].values
        lim = max(0.001, xs.max(), ys.max())
        plt.plot(xs, ys, 'o-', label="Observed value vs predicted value")
        plt.plot([0,lim],[0,lim], '--', color='grey', label="Ideal line")
        plt.xlabel("Predicting risks"); plt.ylabel("Monitoring risks"); plt.title(f"Calibration curve（{tau}mouth）"); plt.legend()
        save_fig(Path(outdir)/f"F7_calibration_{tau}m.png")

        thresh = np.linspace(0.01, 0.80, CONFIG["_DCA_POINTS"])
        NB_model, NB_all, NB_none = [], [], []
        y_arr = yb.values
        for pt in thresh:
            pred_pos = (r >= pt).astype(int)
            tp = np.sum((pred_pos==1) & (y_arr==1))
            fp = np.sum((pred_pos==1) & (y_arr==0))
            N  = len(y_arr)
            nb = (tp/N) - (fp/N)*(pt/(1-pt))
            tp_all = np.sum(y_arr==1); fp_all = np.sum(y_arr==0)
            nb_all = (tp_all/N) - (fp_all/N)*(pt/(1-pt))
            NB_model.append(nb); NB_all.append(nb_all); NB_none.append(0.0)
        # ===== PATCH-6: 规则≥k 曲线（k=10/20/30/40），用 KM 得到组内事件概率当作个体预测概率 =====
        def _km_event_prob(mask):
            sub = data.loc[m & mask, ["T","E"]].dropna()
            if sub.empty: return np.nan
            km = KaplanMeierFitter()
            km.fit(sub["T"].values, sub["E"].values.astype(int))
            try:
                S_t = km.survival_function_at_times(tau).values[-1]
            except Exception:
                S_t = float(km.predict(tau))
            return float(1.0 - S_t)

        # 为每个 k 生成一条 DCA 曲线
        rule_curves = {}  # {name: [nb(pt) ...]}
        # —— 稳健地获取 ELN 向量：优先用 _ELN_source，其次尝试 data['ELN']，都失败就跳过规则曲线 ——
        eln_vals_all = None
        if "_ELN_source" in data.columns:
            eln_vals_all = pd.to_numeric(data.loc[m, "_ELN_source"], errors="coerce").values
        elif "ELN" in data.columns:
            eln_vals_all = pd.to_numeric(data.loc[m, "ELN"], errors="coerce").values

        rule_curves = {}  # {name: [nb(pt) ...]}

        def _km_event_prob(mask_full_bool_index):
            """在 data 上根据布尔索引计算 τ 月事件概率（1 - S(τ)）"""
            sub = data.loc[mask_full_bool_index, ["T", "E"]].dropna()
            if sub.empty:
                return np.nan
            km = KaplanMeierFitter()
            km.fit(sub["T"].values, sub["E"].astype(int).values)
            try:
                S_t = km.survival_function_at_times(tau).values[-1]
            except Exception:
                S_t = float(km.predict(tau))
            return float(1.0 - S_t)

        if eln_vals_all is not None:
            # 建一个对齐到 data.index 的完整布尔索引容器
            base_mask = pd.Series(False, index=data.index)
            base_mask.loc[m.index[m]] = True  # 让 True 只落在 m==True 的位置上

            for k in [10, 20, 30, 40]:
                high_arr = (eln_vals_all >= k)
                # 把 high_arr 填回完整索引（只在 m==True 的位置上赋值）
                high_mask = base_mask.copy()
                high_mask.loc[m.index[m]] = high_arr

                p_high = _km_event_prob(high_mask)
                p_low = _km_event_prob(~high_mask & base_mask)
                if not (np.isfinite(p_high) and np.isfinite(p_low)):  # 任意失败则跳过该规则
                    continue

                # 个体“预测概率”向量（规则≥k）：m==True 的位置按高/低组给概率，其它位置忽略
                r_rule = np.where(high_arr, p_high, p_low)

                NB_rule = []
                for pt in thresh:
                    pred_pos = (r_rule >= pt).astype(int)
                    tp = np.sum((pred_pos == 1) & (y_arr == 1))
                    fp = np.sum((pred_pos == 1) & (y_arr == 0))
                    N = len(y_arr)
                    nb = (tp / N) - (fp / N) * (pt / (1 - pt))
                    NB_rule.append(nb)
                rule_curves[f"Rule≥{k}"] = NB_rule
        else:
            print("⚠️ Failed to obtain ELN (_ELN_source/ELN both absent), skipping rules ≥k DCA curve; plot model curve only.")

        set_chinese_font()
        plt.figure(figsize=(6,4))
        plt.plot(thresh, NB_model, label="Model (Cox-based hazard)")
        plt.plot(thresh, NB_all,  label="Treat-All", ls="--")
        plt.plot(thresh, NB_none, label="Treat-None", ls="-.")
        # 叠加规则曲线
        for name, nb_list in rule_curves.items():
            plt.plot(thresh, nb_list, label=name)
        plt.xlabel("Threshold probability"); plt.ylabel("Net profit"); plt.title(f"DCA curve（{tau}mouth）")
        plt.legend()
        save_fig(Path(outdir)/f"F8_DCA_with_rules_{tau}m.png")  # 新文件名，保留原图


# ============ LNR 补充 ============
def binary_at_time_labels(df, usable, tau):
    d = df.loc[usable, ["SurvM","Event"]].copy()
    y = np.where((d["Event"]==1) & (d["SurvM"]<=tau), 1, np.where(d["SurvM"]>tau, 0, np.nan))
    return pd.Series(y, index=d.index[usable])

def lnr_block(df, usable, outdir):
    print("🧩 LNR Supplementary Analysis (KM + 1/3-Year ROC Approximation) ...")
    ensure_dir(outdir)
    d = df.loc[usable, ["SurvM","Event","LNR"]].dropna()
    if d.shape[0] < 100: return

    def _lnr_group_scalar(x: float) -> str:
        if pd.isna(x):
            return np.nan
        if x == 0:
            return "0"
        return "0-0.2" if (0 < x <= 0.2) else ">0.2"

    d["LNR_grp"] = d["LNR"].map(_lnr_group_scalar)
    d["LNR_grp"] = pd.Categorical(d["LNR_grp"], categories=["0","0-0.2",">0.2"], ordered=True)


    set_chinese_font()
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for name, sub in d.groupby("LNR_grp", observed=False):
        kmf.fit(sub["SurvM"], sub["Event"], label=f"{name} (n={len(sub)})")
        kmf.plot(ci_show=False)
    plt.xlabel("Follow-up period (months)"); plt.ylabel("Overall survival rate"); plt.title("KM curves for different LNR groups")
    save_fig(Path(outdir)/"F9_KM_LNR_groups.png")

    for tau in [12, 36, 60]:
        y = binary_at_time_labels(d, d.index==d.index, tau).dropna()
        idx = y.index
        r = d.loc[idx, "LNR"].values
        auc = roc_auc_score(y.values, r)
        fpr, tpr, _ = roc_curve(y.values, r)
        set_chinese_font()
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'--',color='grey')
        plt.xlabel("1-Specificity"); plt.ylabel("Sensitivity"); plt.title(f"LNR Approximation ROC（{tau}mouth）"); plt.legend()
        save_fig(Path(outdir)/f"F10_ROC_LNR_{tau}m.png")

# ===== PATCH-5: 年份分层 + 温莎化 =====
def year_strata_and_winsor_sensitivity(df, year_col="Year of diagnosis", eln_col="ELN", outdir=None):
    d = df.copy(); outdir = Path(outdir or CONFIG["outdir"])/"sensitivity_year_winsor"; outdir.mkdir(parents=True, exist_ok=True)
    # 年份分段（按需调整分段）
    if year_col in d.columns:
        y = _to_num_series(d[year_col])
        bins = [1997, 2007, 2012, 2017, 2025]; labels = ["2000-2007","2008-2012","2013-2017","2018+"]
        d["_year_band"] = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
        for lb in d["_year_band"].dropna().unique():
            sub = d[d["_year_band"]==lb]
            if len(sub) < 100: continue
            plot_eln_hist_density(sub, eln_col=eln_col, out_path=outdir/f"F1_ELN_hist_density_year_{lb}.png")
            print(f"[Sensitivity - Vintage] {lb} n={len(sub)}")
    else:
        print("[Sensitivity] No year column found; skipping year stratification.")

    # 温莎化 ELN≤60：仅示意一张直方+密度；如需全套结果可按主流程复制
    d["ELN_w"] = np.clip(_to_num_series(d[eln_col]), a_min=None, a_max=60)
    tmp = d.drop(columns=[eln_col], errors="ignore").rename(columns={"ELN_w": eln_col})
    plot_eln_hist_density(tmp, eln_col=eln_col, out_path=outdir / "F1_ELNw_hist_density.png")
    return d

#生成 T1 基线表（含总体 & ELN≥10/<10 分组的 SMD）与**（可选）F1 流程图**
def _smd_cont(x1, x0):
    m1, s1 = np.nanmean(x1), np.nanstd(x1, ddof=1)
    m0, s0 = np.nanmean(x0), np.nanstd(x0, ddof=1)
    sd = np.sqrt((s1**2 + s0**2)/2.0)
    return 0.0 if (sd==0 or not np.isfinite(sd)) else (m1-m0)/sd

def _smd_cat(s1, s0):
    # one-vs-rest SMD 的最大值（保守）
    levels = sorted(set(list(pd.Series(s1).dropna().unique()) + list(pd.Series(s0).dropna().unique())))
    max_smd = 0.0
    for lv in levels:
        p1 = np.mean(pd.Series(s1)==lv); p0 = np.mean(pd.Series(s0)==lv)
        p  = (p1+p0)/2.0; sd = np.sqrt(p*(1-p))
        smd = 0.0 if (sd==0 or not np.isfinite(sd)) else (p1-p0)/sd
        max_smd = max(max_smd, abs(smd))
    return max_smd

def make_table1_and_flowchart(df, usable, outdir):
    from pathlib import Path
    import matplotlib.pyplot as plt

    d = df.loc[usable].copy()
    # 分组：最低合格线（与你论文一致）
    treat = (d["ELN"]>=10).astype(int)
    d["ELN_ge10"] = treat

    rows = []
    def row_var(name, overall, trt, ctrl, smd):
        rows.append({"Variable": name, "Overall": overall, "ELN≥10": trt, "ELN<10": ctrl, "SMD": smd})

    # 连续型：Age, ELN, POS, LNR, SurvM
    for v, fmt in [("Age","{:.1f}±{:.1f}"),
                   ("ELN","{:.0f} ({:.0f}-{:.0f})"),
                   ("POS","{:.0f} ({:.0f}-{:.0f})"),
                   ("LNR","{:.3f} ({:.3f}-{:.3f})"),
                   ("SurvM","{:.0f} ({:.0f}-{:.0f})")]:
        if v in d.columns:
            x = pd.to_numeric(d[v], errors="coerce")
            ov = fmt.format(np.nanmean(x), np.nanstd(x, ddof=1)) if "±" in fmt else fmt.format(
                 np.nanmedian(x), np.nanquantile(x,0.25), np.nanquantile(x,0.75))
            x1, x0 = x[d["ELN_ge10"]==1], x[d["ELN_ge10"]==0]
            tr = fmt.format(np.nanmean(x1), np.nanstd(x1, ddof=1)) if "±" in fmt else fmt.format(
                 np.nanmedian(x1), np.nanquantile(x1,0.25), np.nanquantile(x1,0.75))
            ct = fmt.format(np.nanmean(x0), np.nanstd(x0, ddof=1)) if "±" in fmt else fmt.format(
                 np.nanmedian(x0), np.nanquantile(x0,0.25), np.nanquantile(x0,0.75))
            smd = _smd_cont(x1, x0) if "±" in fmt else _smd_cont(x1, x0)
            row_var(v, ov, tr, ct, round(smd,3))

    # 分类型：Sex, Primary_part, Stage, SurgeryExtent, chemo_bin, radiation_bin
    for v in ["Sex","Primary_part","Stage","SurgeryExtent","chemo_bin","radiation_bin"]:
        if v in d.columns:
            s = d[v]
            ov = "; ".join([f"{k} {v2}({v2/len(s):.1%})" for k,v2 in s.value_counts(dropna=False).items()][:6])
            s1 = s[d["ELN_ge10"]==1]; s0 = s[d["ELN_ge10"]==0]
            tr = "; ".join([f"{k} {v2}({v2/len(s1):.1%})" for k,v2 in s1.value_counts(dropna=False).items()][:6])
            ct = "; ".join([f"{k} {v2}({v2/len(s0):.1%})" for k,v2 in s0.value_counts(dropna=False).items()][:6])
            smd = _smd_cat(s1, s0)
            row_var(v, ov, tr, ct, round(smd,3))

    t1 = pd.DataFrame(rows)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    t1.to_excel(out/"T1_Baseline Table.xlsx", index=False)

    # （可选）极简流程图：统计各步样本数
    N0 = len(df)
    N1 = int((df["SurvM"].notna() & df["Event"].notna() & (df["SurvM"]>0)).sum())
    m_flow = df["SurvM"].notna() & df["Event"].notna() & (df["SurvM"]>0)
    if "Stage" in df.columns and CONFIG.get("drop_stage_other", True):
        m_flow = m_flow & (~df["Stage"].astype(str).str.strip().eq("Other"))
    if "SurgeryExtent" in df.columns and CONFIG.get("drop_surgery_other", True):
        m_flow = m_flow & (~df["SurgeryExtent"].astype(str).str.strip().str.lower().eq("other"))
    N2 = int(m_flow.sum())

# ============ 主程序 ============
def main():
    np.random.seed(CONFIG["random_state"])
    ensure_dir(Path(CONFIG["outdir"]))

    # 快/全挡派生
    FAST = CONFIG.get("fast_mode", True)
    CONFIG["_BOOT_CINDEX"] = CONFIG["bootstrap_n_cindex_fast"] if FAST else CONFIG["bootstrap_n_cindex_full"]
    CONFIG["_BOOT_CP"]     = CONFIG["bootstrap_n_cp_fast"]     if FAST else CONFIG["bootstrap_n_cp_full"]
    CONFIG["_CP_GRID"]     = CONFIG["cp_grid_fast"]            if FAST else CONFIG["cp_grid_full"]
    CONFIG["_DCA_POINTS"]  = CONFIG["dca_points_fast"]         if FAST else CONFIG["dca_points_full"]
    CONFIG["_STRAT_METHS"] = CONFIG["strategy_methods_fast"]   if FAST else CONFIG["strategy_methods_full"]
    CONFIG["_STRAT_CUTS"]  = CONFIG["strategy_cuts_fast"]      if FAST else CONFIG["strategy_cuts_full"]
    print(f"⏩ Operating Mode：{'Quick Shift' if FAST else 'Full Block'} | C-index self-service={CONFIG['_BOOT_CINDEX']} | CP Self-Service={CONFIG['_BOOT_CP']} | "
          f"CP candidate number={len(CONFIG['_CP_GRID'])} | DCA score={CONFIG['_DCA_POINTS']} | Strategy={CONFIG['_STRAT_METHS']}×Threshold{CONFIG['_STRAT_CUTS']}")

    # 读取
    df, found, usable = load_and_prepare()
    # NEW: 生成 T1 & F1
    make_table1_and_flowchart(df, usable, CONFIG["outdir"])
    with open(Path(CONFIG["outdir"])/"meta.json","w",encoding="utf-8") as f:
        json.dump({"rows": int(df.shape[0]), "cols": int(df.shape[1]),
                   "usable": int(usable.sum()), "found_cols": found}, f, ensure_ascii=False, indent=2)

    # 基线 & 流程图（保持你现有版本）
    print("📄 Generate baseline table (Table 1) ...")

    # 分箱 + 步长=1（最新版口径）
    eln_binning_and_plots(df, usable, CONFIG["outdir"])

    # 阈值搜索 + KM
    best_cut, _ = eln_threshold_search_and_km(df, usable, CONFIG["outdir"])
    print(f"🏁 Optimal ELN Threshold（log-rank）：{best_cut}")

    # Cox + PH + C-index
    cph, cox_data = cox_fit_and_plots(df, usable, CONFIG["outdir"])

    # RCS（可开关）
    if CONFIG.get("do_rcs", True):
        rcs_plot_ELN(df, usable, CONFIG["outdir"], cox_data)

    # 变化点
    c_hat, ci, _ = changepoint_search(df, usable, CONFIG["outdir"])
    print(f"🪓 Point estimation of change：{c_hat}，95%CI={ci}")

    # 策略化
    strategies_block(df, usable, CONFIG["outdir"])

    # 校准 & DCA
    calibration_and_dca(cph, cox_data, CONFIG["outdir"])

    # LNR 补充
    lnr_block(df, usable, CONFIG["outdir"])
    # 敏感性分析：诊断年份分层 + 温莎化
    _ = year_strata_and_winsor_sensitivity(df, year_col="Year of diagnosis", eln_col="ELN", outdir=CONFIG["outdir"])

    # 清单
    with open(Path(CONFIG["outdir"])/"INDEX_Product List.txt", "w", encoding="utf-8") as f:
        f.write(
            "【Principal Export List】\n"
            "- ELN_bins_overall.xlsx / by_stage.xlsx / by_surgery.xlsx / ELN_bins_and_strata.xlsx\n"
            "- KM_ELN_bins_overall.png / KM_ELN_bins_Stage_*.png / KM_ELN_bins_Surgery_*.png\n"
            "- Barchart_Median_by_ELNbin_overall.png / Line_Median_by_ELNbin_overall.png / ...\n"
            "- F4_ELN_step1_medianOS_KM.png\n"
            "- F2_KM_ELN_cut*.png / T2a_top10.xlsx / T2b_all.xlsx\n"
            "- F5_RCS_HR_ELN.png / F5b_Changepoint_search.png + T6_*.xlsx/json\n"
            "- F6_ForestPlot_Cox.png + T3_Cox_summary.xlsx + T3_fit_attempts.json + T3_dropped_columns.json\n"
            "- T5_strategy_DeltaRMST.xlsx + F7_KM_weighted_cut*.png\n"
            "- F7_calibration_12m/36m/60m.png + T7_calibration_*.xlsx\n"
            "- F8_DCA_12m/36m/60m.png\n"
            "- F9_KM_LNR_groups.png + F10_ROC_LNR_12m/36m.png\n"
        )
    print("✅ Full process completed, product catalogue：", Path(CONFIG["outdir"]).resolve())

if __name__ == "__main__":
    main()
