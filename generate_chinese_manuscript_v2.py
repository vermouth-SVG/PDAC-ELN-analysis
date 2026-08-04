# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import part1_seer_pipeline as p1


ROOT = Path(__file__).resolve().parent
PART1_DIR = ROOT / "outputs_part1_seer_analysis_full"
PART2_DIR = ROOT / "outputs_part2_anatomical_evidence"
PART1_TABLES = PART1_DIR / "tables" / "Part1_SEER_Tables.xlsx"
PART2_TABLES = PART2_DIR / "tables" / "Part2_Anatomical_Evidence_Tables_Submission.xlsx"
RMST_BOOT = ROOT / "rmst_bootstrap_results.csv"
RMST_P = ROOT / "rmst_bootstrap_pnorm.csv"
OUTPUT_DOCX = ROOT / "中文初稿v2.docx"


def set_run_font(run, size: float = 12, bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    run.font.size = Pt(size)
    run.bold = bold


def style_paragraph(
    paragraph,
    *,
    size: float = 12,
    bold: bool = False,
    align: WD_PARAGRAPH_ALIGNMENT = WD_ALIGN_PARAGRAPH.JUSTIFY,
    first_indent: bool = True,
    line_spacing: float = 1.5,
    space_after_pt: float = 0,
) -> None:
    fmt = paragraph.paragraph_format
    fmt.first_line_indent = Pt(24) if first_indent else Pt(0)
    fmt.line_spacing = line_spacing
    fmt.space_after = Pt(space_after_pt)
    fmt.space_before = Pt(0)
    paragraph.alignment = align
    for run in paragraph.runs:
        set_run_font(run, size=size, bold=bold)


def add_paragraph(
    doc: Document,
    text: str,
    *,
    size: float = 12,
    bold: bool = False,
    align: WD_PARAGRAPH_ALIGNMENT = WD_ALIGN_PARAGRAPH.JUSTIFY,
    first_indent: bool = True,
    line_spacing: float = 1.5,
    space_after_pt: float = 0,
) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    style_paragraph(
        p,
        size=size,
        bold=bold,
        align=align,
        first_indent=first_indent,
        line_spacing=line_spacing,
        space_after_pt=space_after_pt,
    )


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    if level == 1:
        add_paragraph(doc, text, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, first_indent=False, line_spacing=1.2, space_after_pt=6)
    else:
        add_paragraph(doc, text, size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, first_indent=False, line_spacing=1.2, space_after_pt=3)


def add_title_page(doc: Document, title: str) -> None:
    add_paragraph(doc, title, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.2, space_after_pt=8)
    add_paragraph(doc, "作者：____________________", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.2)
    add_paragraph(doc, "单位：____________________", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.2)
    add_paragraph(doc, "通讯作者：____________________", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.2, space_after_pt=12)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    normal.font.size = Pt(12)


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def sci_text(p: float) -> str:
    if p <= 0:
        return "<0.001"
    exp = int(np.floor(np.log10(p)))
    coef = p / (10 ** exp)
    return f"{coef:.2f}×10^{exp}"


def cell_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def translate_table1(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Age": "年龄",
        "Sex": "性别",
        "Female": "女",
        "Male": "男",
        "Stage": "SEER分期",
        "Localized": "局限期",
        "Regional": "区域期",
        "Distant": "远处转移期",
        "Unknown": "分期不明",
        "Primary_part": "肿瘤部位",
        "Body": "体部",
        "Tail": "尾部",
        "SurgeryExtent_clean": "手术范围",
        "Local": "局部切除",
        "Combined": "联合切除",
        "Extended": "扩大切除",
        "Other": "其他",
        "chemo_bin": "化疗",
        "radiation_bin": "放疗",
        "Yes": "是",
        "No": "否",
    }
    out = df.copy()
    first_col = out.columns[0]
    out[first_col] = out[first_col].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
    return out


def translate_table3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().rename(
        columns={
            "Approach family": "手术入路家族",
            "Cohort": "队列",
            "n": "样本量",
            "Design": "研究设计",
            "Reported ELN": "原始ELN报告值",
            "Harmonized mean±SD": "统一化均值±标准差",
            "P(ELN>=12)": "P(ELN≥12)",
            "P(ELN>=14)": "P(ELN≥14)",
            "P(ELN>=21)": "P(ELN≥21)",
        }
    )
    family_map = {
        "Standard / conventional DP": "标准/常规DP",
        "RAMPS": "RAMPS",
        "Posterior / artery-first spectrum": "后方/动脉优先谱系",
    }
    design_map = {
        "Location-stratified cohort": "按肿瘤部位分层队列",
        "PSM comparator arm": "倾向评分匹配对照臂",
        "Randomized arm": "随机试验臂",
        "Dual-institution comparator arm": "双中心比较对照臂",
        "Comparative RAMPS cohort": "RAMPS比较队列",
        "Single-arm Lap-RAMPS": "单臂腹腔镜RAMPS",
        "PSM RAMPS cohort": "倾向评分匹配RAMPS队列",
        "Single-arm posterior-route series": "单臂后方入路系列",
    }
    out["手术入路家族"] = out["手术入路家族"].map(lambda x: family_map.get(x, x))
    out["研究设计"] = out["研究设计"].map(lambda x: design_map.get(x, x))
    return out


def translate_table4(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().rename(
        columns={
            "Approach family": "手术入路家族",
            "Context": "适用场景",
            "#10/#11/#18": "#10/#11/#18站",
            "#9": "#9站",
            "#14": "#14站",
            "Posterior plane": "后腹膜平面",
            "Key references": "代表性参考文献",
        }
    )
    family_map = {
        "Standard / conventional DP": "标准/常规DP",
        "RAMPS": "RAMPS",
        "Posterior / artery-first spectrum": "后方/动脉优先谱系",
    }
    context_map = {
        "Body/proximal tail standard DP": "胰体/近侧尾部常规DP",
        "Distal-tail standard DP": "远侧尾部常规DP",
        "Comparative/open-or-mixed RAMPS": "比较性/开放或混合RAMPS",
        "Left-posterior / artery-first variants": "左后方/动脉优先变式",
    }
    value_map = {
        "Routine": "常规可及",
        "Selective": "选择性可及",
        "Usually low-yield": "通常收益低",
        "Selective to enhanced": "选择性至增强可及",
        "Enhanced": "增强可及",
        "Direct": "直接可及",
        "Limited": "有限",
    }
    out["手术入路家族"] = out["手术入路家族"].map(lambda x: family_map.get(x, x))
    out["适用场景"] = out["适用场景"].map(lambda x: context_map.get(x, x))
    for col in ["#10/#11/#18站", "#9站", "#14站", "后腹膜平面"]:
        out[col] = out[col].map(lambda x: value_map.get(x, x))
    return out


def translate_table5(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().rename(
        columns={
            "Approach family": "手术入路家族",
            "Estimable cohorts": "可估计队列数",
            "Combined n": "合并样本量",
            "Combined mean±SD": "合并均值±标准差",
            "P(ELN≥12)": "P(ELN≥12)",
            "P(ELN≥14)": "P(ELN≥14)",
            "P(ELN≥21)": "P(ELN≥21)",
            "RR vs Standard (ELN≥21)": "相对风险（vs 标准DP，ELN≥21）",
            "Recommendation": "建议定位",
            "Interpretation": "临床解释",
        }
    )
    family_map = {
        "Standard / conventional DP": "标准/常规DP",
        "RAMPS": "RAMPS",
        "Posterior / artery-first spectrum": "后方/动脉优先谱系",
    }
    rec_map = {
        "Baseline reference": "基线参照",
        "Preferred escalation": "优先升级方案",
        "Trend-supported, low-certainty": "趋势支持、低确定性",
    }
    interp_map = {
        "Standard / conventional DP": "该家族可作为常规基线参照，通常能够支持ELN 12-14枚的达成，但稳定跨越21枚仍较困难。",
        "RAMPS": "该家族ELN分布相对标准DP整体右移，可明显提升跨越ELN=21的概率，但该关联并不等同于已证实的生存因果获益。",
        "Posterior / artery-first spectrum": "现有文献提示后方/动脉优先谱系具有较高ELN获取潜力，但证据稀少、确定性低，尚不能证明其优于标准DP或RAMPS。",
    }
    out["手术入路家族"] = out["手术入路家族"].map(lambda x: family_map.get(x, x))
    out["建议定位"] = out["建议定位"].map(lambda x: rec_map.get(x, x))
    out["临床解释"] = df["Approach family"].map(lambda x: interp_map.get(x, x))
    return out


def build_rmst_table() -> pd.DataFrame:
    points = pd.read_csv(RMST_BOOT)
    pvals = pd.read_csv(RMST_P)
    base = pd.read_excel(PART1_TABLES, sheet_name="SEER_RMST_All_Methods")
    out = base.merge(points[["cut", "method", "tau_m", "CI_low", "CI_high"]], on=["cut", "method", "tau_m"], how="left")
    out = out.merge(pvals[["cut", "method", "tau_m", "p_norm"]], on=["cut", "method", "tau_m"], how="left")
    out["95% CI"] = out.apply(lambda r: f"{r['CI_low']:.2f}～{r['CI_high']:.2f}" if pd.notna(r["CI_low"]) else "", axis=1)
    out["P值"] = out["p_norm"].map(fmt_p)
    out["加权方法"] = out["method"]
    out["ELN阈值"] = out["cut"].map(lambda x: f"ELN≥{int(x)}")
    out["时间截点（月）"] = out["tau_m"].astype(int)
    out["高ELN组RMST（月）"] = out["RMST_high"].map(lambda x: f"{x:.2f}")
    out["低ELN组RMST（月）"] = out["RMST_low"].map(lambda x: f"{x:.2f}")
    out["ΔRMST（月）"] = out["Delta_RMST"].map(lambda x: f"{x:.2f}")
    out = out[
        ["ELN阈值", "加权方法", "时间截点（月）", "高ELN组RMST（月）", "低ELN组RMST（月）", "ΔRMST（月）", "95% CI", "P值"]
    ].sort_values(["ELN阈值", "加权方法", "时间截点（月）"], kind="stable")
    return out.reset_index(drop=True)


def add_dataframe_table(doc: Document, df: pd.DataFrame, *, font_size: float = 9.5) -> None:
    table = doc.add_table(rows=df.shape[0] + 1, cols=df.shape[1])
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for j, col in enumerate(df.columns):
        table.cell(0, j).text = str(col)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            table.cell(i + 1, j).text = cell_text(df.iat[i, j])

    for i, row in enumerate(table.rows):
        for j, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 or j != 0 else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.first_line_indent = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                for run in p.runs:
                    set_run_font(run, size=font_size, bold=(i == 0))


def add_figure(doc: Document, image_path: Path, caption: str, *, width_in: float = 6.1) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_in))
    style_paragraph(p, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.0, space_after_pt=3)
    add_paragraph(doc, caption, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.15, space_after_pt=6)


def add_reference(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.add_run(text)
    p.paragraph_format.left_indent = Pt(0)
    p.paragraph_format.first_line_indent = Pt(-18)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    for run in p.runs:
        set_run_font(run, size=10.5)


def load_metrics() -> dict[str, object]:
    config = p1.AnalysisConfig(excel_path="Pancreatic_Body_Tail_ALL.xlsx", sheet_name="export", outdir=str(PART1_DIR))
    analysis_df, summary = p1.load_and_prepare_dataset(config)

    analysis_df = analysis_df.copy()
    analysis_df["ELN_group12"] = np.where(analysis_df["ELN"] >= 12, "ELN≥12", "ELN<12")
    event_pct = summary["event_n"] / summary["analysis_n"] * 100

    km = KaplanMeierFitter()
    medians = {}
    for grp in ["ELN≥12", "ELN<12"]:
        sub = analysis_df.loc[analysis_df["ELN_group12"] == grp]
        km.fit(sub["SurvM"], sub["Event"])
        medians[grp] = float(km.median_survival_time_)

    lr = logrank_test(
        analysis_df.loc[analysis_df["ELN_group12"] == "ELN≥12", "SurvM"],
        analysis_df.loc[analysis_df["ELN_group12"] == "ELN<12", "SurvM"],
        event_observed_A=analysis_df.loc[analysis_df["ELN_group12"] == "ELN≥12", "Event"],
        event_observed_B=analysis_df.loc[analysis_df["ELN_group12"] == "ELN<12", "Event"],
    )

    study_summary = pd.read_excel(PART1_TABLES, sheet_name="Study_Summary")
    summary_map = dict(zip(study_summary["Item"], study_summary["Value"]))
    eln_cox = pd.read_excel(PART1_TABLES, sheet_name="SEER_ELN_Cox_Model")
    eln_ph = pd.read_excel(PART1_TABLES, sheet_name="ELN_PH_Test")
    pos_transition = pd.read_excel(PART1_TABLES, sheet_name="POS_Step_Effect")
    pos_rmst = pd.read_excel(PART1_TABLES, sheet_name="POS_Delta_RMST")
    part2_ref_summary = pd.read_excel(PART2_TABLES, sheet_name="Reference_Summary")
    part2_study_catalog = pd.read_excel(PART2_TABLES, sheet_name="Study_Catalog")

    eln_row = eln_cox.loc[eln_cox["covariate"] == "ELN_per5"].iloc[0]
    eln_ph_row = eln_ph.loc[eln_ph["covariate"] == "ELN_per5"].iloc[0]
    pos_row = pos_transition.loc[pos_transition["group"] == "POS>=1"].iloc[0]
    ref_rows = part2_ref_summary.loc[part2_ref_summary["level"] == "Family"].copy()
    ref_rows["family"] = ref_rows["family"].astype(str)
    standard_row = ref_rows.loc[ref_rows["family"] == "Standard / conventional DP"].iloc[0]
    ramps_row = ref_rows.loc[ref_rows["family"] == "RAMPS"].iloc[0]
    posterior_row = ref_rows.loc[ref_rows["family"] == "Posterior / artery-first spectrum"].iloc[0]

    stage_counts = analysis_df["Stage"].astype(str).value_counts()
    site_counts = analysis_df["Primary_part"].astype(str).value_counts()
    rmst_main = build_rmst_table()

    pos_rmst_60 = pos_rmst.loc[pos_rmst["tau_m"] == 60].copy()
    pos_rmst_60_map = dict(zip(pos_rmst_60["contrast"], pos_rmst_60["Delta_RMST"]))

    return {
        "analysis_df": analysis_df,
        "summary": summary,
        "summary_map": summary_map,
        "event_pct": event_pct,
        "median_os_high12": medians["ELN≥12"],
        "median_os_low12": medians["ELN<12"],
        "logrank_p12": float(lr.p_value),
        "stage_counts": stage_counts,
        "site_counts": site_counts,
        "eln_row": eln_row,
        "eln_ph_row": eln_ph_row,
        "pos_row": pos_row,
        "part2_studies_n": int(len(part2_study_catalog)),
        "standard_row": standard_row,
        "ramps_row": ramps_row,
        "posterior_row": posterior_row,
        "rmst_main": rmst_main,
        "pos_rmst_60_map": pos_rmst_60_map,
    }


def main() -> None:
    metrics = load_metrics()

    table1 = translate_table1(pd.read_excel(PART1_TABLES, sheet_name="Table1_Descriptive_Stats"))
    table3 = translate_table3(pd.read_excel(PART2_TABLES, sheet_name="Table3_Approach_Comparison"))
    table4 = translate_table4(pd.read_excel(PART2_TABLES, sheet_name="Table4_Anatomical_Access"))
    table5 = translate_table5(pd.read_excel(PART2_TABLES, sheet_name="Table5_Anchor_Attainment"))
    table2 = metrics["rmst_main"]

    summary = metrics["summary"]
    summary_map = metrics["summary_map"]
    event_pct = metrics["event_pct"]
    eln_row = metrics["eln_row"]
    eln_ph_row = metrics["eln_ph_row"]
    pos_row = metrics["pos_row"]
    standard_row = metrics["standard_row"]
    ramps_row = metrics["ramps_row"]
    posterior_row = metrics["posterior_row"]
    stage_counts = metrics["stage_counts"]
    site_counts = metrics["site_counts"]
    pos_rmst_60_map = metrics["pos_rmst_60_map"]

    refs = [
        "1. Stoop TF, et al. Pancreatic cancer. Lancet. 2025;405:1182-1202.",
        "2. Siegel RL, Kratzer TB, Giaquinto AN, Sung H, Jemal A. Cancer statistics, 2025. CA Cancer J Clin. 2025;75:10-45.",
        "3. Chen Z, et al. Prognostic and microenvironmental analysis of pancreatic cancer by tumor location. Discov Oncol. 2025;16:2283.",
        "4. Slidell MB, Chang DC, Cameron JL, et al. Impact of total lymph node count and lymph node ratio on staging and survival after pancreatectomy for pancreatic adenocarcinoma: a large, population-based analysis. Ann Surg Oncol. 2008;15(1):165-174.",
        "5. Huang L, Jansen L, Balavarca Y, et al. Significance of examined lymph node number in accurate staging and long-term survival in resected stage I-II pancreatic cancer-more is better? A large international population-based cohort study. Ann Surg. 2021;274:e554-e563.",
        "6. Gonen M, Schrag D, Weiser MR. Nodal staging score: a tool to assess adequate staging of node-negative colon cancer. J Clin Oncol. 2009;27:6166-6171.",
        "7. Yoon SJ, Hong SS, Park B, et al. Optimal number of lymph nodes retrieved to lower the risk of false N0 for patients with pancreatic cancer undergoing curative surgery. Ann Surg Oncol. 2025;32:8894-8902.",
        "8. National Comprehensive Cancer Network. NCCN Clinical Practice Guidelines in Oncology: Pancreatic Adenocarcinoma. Version 3.2024.",
        "9. Tol JA, Gouma DJ, Bassi C, et al. Definition of a standard lymphadenectomy in surgery for pancreatic ductal adenocarcinoma: a consensus statement by the ISGPS. Surgery. 2014;156:591-600.",
        "10. Malleo G, Maggino L, Ferrone CR, et al. Number of examined lymph nodes and nodal status assessment in distal pancreatectomy for body/tail ductal adenocarcinoma. Ann Surg. 2019;270(6):1138-1146.",
        "11. Li YF, Xiang YC, Zhang QQ, Wang WL. Impact of examined lymph node count on prognosis in patients with lymph node-negative pancreatic body/tail ductal adenocarcinoma. J Gastrointest Oncol. 2021;12(4):1647-1656.",
        "12. Xu H, Huang H, Yan A, et al. SEER-based hypothesis-generating research for minimum lymph node evaluation in early-stage pancreatic ductal adenocarcinoma patients. Front Surg. 2025;12:1605726.",
        "13. Minagawa N, Sato N, Mori Y, et al. Optimal lymph node dissection for pancreatic body and tail cancer based on tumor location. Ann Surg Oncol. 2022;29(4):2414-2424.",
        "14. Strasberg SM, Drebin JA, Linehan D. Radical antegrade modular pancreatosplenectomy. Surgery. 2003;133(5):521-527.",
        "15. Abe T, Ohuchida K, Miyasaka Y, et al. Comparison of surgical outcomes between RAMPS and standard retrograde pancreatosplenectomy for left-sided pancreatic cancer. World J Surg. 2016;40(9):2267-2275.",
        "16. Kwon J, Kim K, Chie EK, et al. Radical antegrade modular pancreatosplenectomy versus conventional distal pancreatectomy for left-sided pancreatic cancer: a propensity score-matched analysis. Ann Surg Oncol. 2024;31(8):5234-5242.",
        "17. Yin J, Yang Y, Ma K, et al. Radical antegrade modular pancreatosplenectomy versus conventional distal pancreatectomy for pancreatic body and tail cancer: a propensity-matched study from the CSPO. HPB (Oxford). 2025.",
        "18. Nagai M, Sho M, Satoi S, et al. Left-posterior approach for left-sided pancreatic cancer. J Hepatobiliary Pancreat Sci. 2020;27(12):956-963.",
        "19. Ome Y, Sato N, Hashimoto K, et al. Laparoscopic distal pancreatectomy using caudo-dorsal artery-first approach. Ann Surg Oncol. 2019;26(12):4108-4109.",
        "20. Zhou J, Dou X, Wei W, et al. Higher lymph node burden predicts greater chemotherapy benefit in resected pancreatic ductal adenocarcinoma: evidence from 22,045 patients. Sci Rep. 2026;16:7227.",
        "21. Habib JR, Rompen IF, Kaslow SR, et al. Defining the minimal and optimal thresholds for lymph node resection and examination for IPMN-derived pancreatic cancer. Ann Surg. 2024;282(6):1052-1059.",
        "22. Seo HK, Shin SH, Song KB, et al. Lymph node metastasis patterns of pancreatic tail cancer and the role of extended lymphadenectomy. J Hepatobiliary Pancreat Sci. 2022;29(5):543-551.",
        "23. Sham JG, Guo S, Engelman DW, et al. Radical antegrade modular pancreatosplenectomy versus standard distal pancreatosplenectomy for pancreatic cancer, a dual-institutional analysis. Chin Clin Oncol. 2020;9(4):54.",
        "24. Tanaka M, Heckler M, Mihaljevic AL, et al. Systematic review and meta-analysis of lymph node metastases of resected pancreatic ductal adenocarcinoma. Ann Surg Oncol. 2022;29(1):607-618.",
        "25. Kersik A, et al. The role of non-peripancreatic lymph nodes in the survival of patients with pancreatic cancer of the body and tail: a systematic review and meta-analysis. Cancers (Basel). 2023;15(8):2322.",
        "26. Kim EY, You YK, Kim DG, et al. Initial experience with radical antegrade modular pancreatosplenectomy in a single institution. Ann Surg Treat Res. 2016;91(1):29-36.",
        "27. Larkins K, Rowcroft A, Goh SK, et al. Laparoscopic radical antegrade modular pancreatosplenectomy: a scoping review. ANZ J Surg. 2021;91(7-8):1350-1356.",
        "28. Portelli M, Gauci C, Kobrosliy K, Abela JE. Standard distal pancreatectomy versus radical antegrade modular pancreatosplenectomy: a systematic review and meta-analysis of the literature. Cureus. 2026;18(3):e105617. doi:10.7759/cureus.105617.",
        "29. Conroy T, Hammel P, Hebbar M, et al. FOLFIRINOX or gemcitabine as adjuvant therapy for pancreatic cancer. N Engl J Med. 2018;379:2395-2406.",
    ]

    doc = Document()
    configure_document(doc)
    add_title_page(doc, "胰腺体尾部导管腺癌根治术中最佳送检淋巴结数及手术路径选择：基于SEER的群体研究与解剖证据综合")

    add_heading(doc, "摘要", 1)
    add_paragraph(
        doc,
        "背景：胰腺体尾部导管腺癌（pancreatic ductal adenocarcinoma，PDAC）远端胰腺切除术的最佳送检淋巴结数（examined lymph node，ELN）仍存争议。现行指南建议至少12～15枚，但更高ELN阈值的额外生存获益及不同手术路径达到该阈值的可能性，尚缺乏量化证据。",
        first_indent=False,
    )
    add_paragraph(
        doc,
        f"方法：纳入SEER数据库2000—2020年接受根治性切除的胰腺体尾部PDAC患者{summary['analysis_n']}例。采用log-rank阈值网格搜索、分段Cox变化点分析、限制性样条Cox模型和重叠加权限制平均生存时间（overlap-weighted restricted mean survival time，OW-RMST）分析识别与总体生存相关的ELN阈值。另对{metrics['part2_studies_n']}项已发表研究进行结构化文献综合，比对标准/常规DP、RAMPS和后方/动脉优先技术谱系达到关键ELN阈值的概率。",
        first_indent=False,
    )
    add_paragraph(
        doc,
        f"结果：队列中共记录{summary['event_n']}例死亡（{event_pct:.1f}%）。log-rank网格搜索识别ELN=12为最优二分阈值；分段Cox在ELN=21处识别生存获益变化点（Bootstrap 95% CI：{summary_map['Change point 95% CI lower']:.1f}～{summary_map['Change point 95% CI upper']:.1f}）。调整后B样条显示ELN与死亡风险呈非线性负相关，ELN与死亡风险的关联在约21枚后减弱。多变量Cox模型中，ELN每增加5枚对应HR {eln_row['exp(coef)']:.3f}（95% CI {eln_row['exp(coef) lower 95%']:.3f}～{eln_row['exp(coef) upper 95%']:.3f}，P<0.001），Bootstrap C指数为{summary_map['Bootstrap c-index mean']:.3f}（95% CI {summary_map['Bootstrap c-index 95% CI lower']:.3f}～{summary_map['Bootstrap c-index 95% CI upper']:.3f}）。60个月OW-RMST净获益分别为：ELN≥12组+2.59个月、ELN≥14组+2.31个月、ELN≥21组+2.60个月。文献综合显示，标准/常规DP、RAMPS与后方/动脉优先谱系的合并ELN分布分别为{standard_row['combined_summary']}、{ramps_row['combined_summary']}和{posterior_row['combined_summary']}；达到ELN≥21的估计概率分别为16.5%、40.0%和82.7%，其中后者证据确定性较低。",
        first_indent=False,
    )
    add_paragraph(
        doc,
        "结论：对于胰腺体尾部PDAC，ELN≥12可作为最低质量基准，ELN≥21可作为高收益目标。RAMPS是目前最有文献支撑的提高清扫强度的标准化升级方案，但其总体生存优势仍未被直接证实。",
        first_indent=False,
    )
    add_paragraph(doc, "关键词：胰腺体尾部癌；送检淋巴结数；远端胰腺切除术；RAMPS；SEER数据库；生存分析", first_indent=False)

    add_heading(doc, "1 引言", 1)
    intro_paragraphs = [
        "胰腺导管腺癌（PDAC）是预后最差的实体恶性肿瘤之一，5年总体生存率长期徘徊在低水平[1,2]。胰腺体尾部PDAC约占全部胰腺癌的20%～25%，由于早期缺乏黄疸等梗阻性提示症状，常在局部进展甚至系统播散阶段才被发现[3]。对于可切除病例，远端胰腺切除联合脾切除仍是唯一具根治潜力的治疗手段[4]。",
        "区域淋巴结的充分病理评估是精准N分期的基础，也是辅助治疗决策的重要前提。送检淋巴结数不足会增加假阴性N0（false-negative N0）的概率，使真正存在淋巴结转移的患者被低估分期并可能错失强化辅助治疗机会[5-7]。NCCN建议至少送检12枚淋巴结，ISGPS则对未接受新辅助治疗患者建议不少于15枚[8,9]。",
        "然而，上述推荐主要来自胰头癌和胰十二指肠切除的数据外推，针对胰腺体尾部切除标本的ELN阈值研究结论仍明显分散。既往报道的最优阈值横跨11～21枚[10-13]，提示单一二分切点难以完整描述ELN与生存获益之间的非线性剂量-反应关系。",
        "手术入路是影响ELN获取量的重要可干预因素。RAMPS通过系统暴露后腹膜平面及腹腔干、脾动脉周围淋巴结站，理论上较标准/常规DP可获得更高ELN[14-17]；左后方入路和动脉优先技术同样可能提高深部后腹膜淋巴结站可及性[18,19]。但这些手术路径是否能稳定推动患者达到具有生存意义的ELN阈值，尚缺乏与SEER生存锚值相衔接的整合证据框架。",
        "因此，本研究拟解决两个互补问题：其一，基于SEER大样本人群系统识别胰腺体尾部PDAC切除术后与总体生存相关的ELN阈值；其二，基于已发表文献综合，反向评估不同手术入路家族达到这些关键阈值的概率及其临床定位。",
    ]
    for text in intro_paragraphs:
        add_paragraph(doc, text)

    add_heading(doc, "2 资料与方法", 1)
    add_heading(doc, "2.1 数据来源与研究对象", 2)
    add_paragraph(
        doc,
        f"研究数据来自SEER 18注册数据库的预处理工作表。当前工作表共包含{summary_map['Raw cohort']}例已满足胰腺体尾部部位、组织学类型和手术基本条件的候选病例。进一步排除生存月数≤0者86例及阳性淋巴结记录缺失/无效者2例后，最终纳入{summary['analysis_n']}例进入主分析。所有纳入患者均为2000—2020年诊断的胰腺体部或尾部原发性PDAC，且ELN≥1、POS记录完整。补充图S1展示患者筛选流程。本研究使用去标识化公开数据，依据SEER使用协议免于机构伦理审批。",
    )
    add_heading(doc, "2.2 ELN阈值识别", 2)
    add_paragraph(
        doc,
        "采用三种互补策略识别ELN阈值：1）在候选切点6～35枚范围内进行log-rank阈值网格搜索，以log-rank统计量最大的切点作为最优二分阈值；2）采用分段Cox变化点搜索识别ELN-生存关系斜率发生改变的位置，并以Bootstrap方法评估不确定性；3）构建调整年龄、性别、SEER分期、原发部位、手术范围、化疗和放疗的限制性B样条Cox模型，描述ELN与死亡风险之间的非线性关联。",
    )
    add_heading(doc, "2.3 多变量生存建模与模型评估", 2)
    add_paragraph(
        doc,
        "多变量Cox模型纳入ELN（每5枚）、年龄（每10岁）、性别、SEER分期、原发部位、手术范围、化疗和放疗等协变量。采用Bootstrap C指数评价模型判别能力，并基于Schoenfeld残差检验比例风险假设。为降低单纯HR解释在时间依赖情形下的局限性，本文同步报告RMST结果。",
    )
    add_heading(doc, "2.4 阳性淋巴结（POS）分析", 2)
    add_paragraph(
        doc,
        "构建包含POS阶跃项（POS=0 vs POS≥1）与连续样条项的Cox模型，刻画POS对死亡风险的离散跳跃效应和连续剂量-反应关系。进一步将POS分为0枚、1～2枚、3～4枚和≥5枚，采用重叠加权RMST比较不同阳性负荷层级对预后的影响。",
    )
    add_heading(doc, "2.5 手术入路证据综合", 2)
    add_paragraph(
        doc,
        f"对2016—2026年公开文献进行结构化综述，共纳入{metrics['part2_studies_n']}项研究。按手术路径分为标准/常规DP、RAMPS和后方/动脉优先谱系三类。对文献中报告的ELN中位数、四分位数或范围采用统一公式折算为均值±标准差，再在家族层面合并均值和方差，推导P(ELN≥12)、P(ELN≥14)和P(ELN≥21)。点估计仅反映汇总数据下的阈值实现概率，不等同于患者级重建或因果效应。",
    )
    add_heading(doc, "2.6 统计学分析", 2)
    add_paragraph(doc, "所有分析均在Python 3.10环境中完成，主要依赖lifelines、pandas、numpy和scikit-learn。双侧P<0.05认为差异有统计学意义。")

    add_heading(doc, "3 结果", 1)
    add_heading(doc, "3.1 队列特征", 2)
    add_paragraph(
        doc,
        f"最终纳入{summary['analysis_n']}例胰腺体尾部PDAC切除患者，其中{summary['event_n']}例死亡（{event_pct:.1f}%），{summary['censored_n']}例在末次随访时被截尾。按SEER分期，局限期{stage_counts.get('Localized', 0)}例、区域期{stage_counts.get('Regional', 0)}例、远处转移期{stage_counts.get('Distant', 0)}例、分期不明{stage_counts.get('Unknown', 0)}例；按原发部位，体部{site_counts.get('Body', 0)}例，尾部{site_counts.get('Tail', 0)}例。以ELN=12为切点分组的基线特征见表1。",
    )
    add_paragraph(doc, "表1 按ELN=12分层的研究对象基线特征", size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.15, space_after_pt=3)
    add_dataframe_table(doc, table1, font_size=9)

    add_heading(doc, "3.2 ELN阈值识别与非线性关系", 2)
    add_paragraph(
        doc,
        f"log-rank阈值网格搜索在候选切点6～35枚范围内识别ELN=12为最优二分阈值。按该切点分组后，ELN≥12组与ELN<12组的中位OS分别为{metrics['median_os_high12']:.1f}个月和{metrics['median_os_low12']:.1f}个月，log-rank P={sci_text(metrics['logrank_p12'])}。分段Cox变化点分析则在ELN=21处达到最大似然，对应Bootstrap 95% CI为{summary_map['Change point 95% CI lower']:.1f}～{summary_map['Change point 95% CI upper']:.1f}，提示ELN对生存的边际贡献在约21枚后明显减弱。",
    )
    add_paragraph(
        doc,
        f"B样条非线性HR曲线以全队列ELN中位数{int(metrics['analysis_df']['ELN'].median())}枚为参照，显示HR在低ELN区间快速下降，在约21枚后呈现关联减弱。上述结果共同支持双阈值框架：ELN=12可视为最低质量基准，而ELN=21更适合作为高收益目标而非刚性界值。",
    )
    add_figure(doc, PART1_DIR / "figures" / "Figure_1_SEER_ELN_Overview.png", "图1 SEER队列ELN分布与生存概览。A：ELN分布直方图；B：按细分ELN区间绘制的Kaplan-Meier生存曲线；C：不同ELN分箱的中位OS；D：按精确ELN值平滑后的中位OS曲线。高ELN尾部分箱及较大ELN值处的精确ELN平滑结果应作为描述性结果解释，因为高ELN计数处样本量较小。")
    add_figure(doc, PART1_DIR / "figures" / "Figure_2_SEER_ELN_Modeling.png", f"图2 ELN阈值的多方法建模。A：log-rank阈值剖面；B：分段Cox变化点剖面，红色虚线为ELN=21，阴影区为Bootstrap 95% CI（{summary_map['Change point 95% CI lower']:.1f}～{summary_map['Change point 95% CI upper']:.1f}）；C：调整后B样条HR曲线，显示约21枚后关联减弱；D：多变量Cox森林图。")

    add_heading(doc, "3.3 多变量Cox模型", 2)
    add_paragraph(
        doc,
        f"多变量Cox模型显示，ELN每增加5枚对应死亡风险下降约3.6%（HR {eln_row['exp(coef)']:.3f}，95% CI {eln_row['exp(coef) lower 95%']:.3f}～{eln_row['exp(coef) upper 95%']:.3f}，P<0.001）。模型Bootstrap C指数为{summary_map['Bootstrap c-index mean']:.3f}（95% CI {summary_map['Bootstrap c-index 95% CI lower']:.3f}～{summary_map['Bootstrap c-index 95% CI upper']:.3f}），提示具有中等判别能力。远处转移、年龄增加和POS阳性是最强的不良预后因素。",
    )
    add_paragraph(
        doc,
        f"Schoenfeld残差检验显示ELN本身未见显著比例风险违背（P={eln_ph_row['p']:.3f}），但分期和化疗等协变量存在一定时间依赖性，因此本文将Cox模型与RMST结果并行解释，以减少单一HR框架的偏倚。",
    )

    add_heading(doc, "3.4 RMST分析", 2)
    add_paragraph(
        doc,
        "OW-RMST结果进一步量化了ELN阈值带来的绝对生存获益。在60个月时间截点，ELN≥12、ELN≥14和ELN≥21对应的OW-RMST净获益分别为+2.59个月（95% CI 1.33～3.92，P<0.001）、+2.31个月（95% CI 0.87～3.51，P<0.001）和+2.60个月（95% CI 1.13～4.01，P<0.001）。12个月和36个月时，各阈值获益方向总体一致，但ELN≥21在12个月时效应量较小。IPTW敏感性分析方向一致，见表2。",
    )
    add_paragraph(doc, "表2 不同ELN阈值在12、36、60个月的加权RMST差异", size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.15, space_after_pt=3)
    add_dataframe_table(doc, table2, font_size=9)

    add_heading(doc, "3.5 POS分析", 2)
    add_paragraph(
        doc,
        f"在全部{summary['analysis_n']}例具有完整POS信息的患者中，从POS=0到POS≥1存在明显风险阶跃效应（HR {pos_row['HR']:.3f}，95% CI {pos_row['HR_lower']:.3f}～{pos_row['HR_upper']:.3f}，P<0.001）。在POS>0患者中，连续样条曲线提示风险比随阳性淋巴结数目增加而单调升高。",
    )
    add_paragraph(
        doc,
        f"截断后的KM曲线显示早期分离，而长期绝对差异由OW-RMST量化。与POS=0相比，60个月OW-RMST净差异分别为：POS=1～2枚{pos_rmst_60_map['POS=1-2 vs POS=0']:.2f}个月、POS=3～4枚{pos_rmst_60_map['POS=3-4 vs POS=0']:.2f}个月、POS≥5枚{pos_rmst_60_map['POS>=5 vs POS=0']:.2f}个月，提示阳性负荷越高，长期绝对生存损失越明显。",
    )
    add_figure(doc, PART1_DIR / "figures" / "Figure_3_SEER_POS_Analysis.png", "图3 POS分析。A：POS=0至POS≥1的风险阶跃效应；B：POS连续样条HR曲线；C：按POS分层的截断Kaplan-Meier生存曲线，显示早期分离；D：不同POS层级相对POS=0的加权ΔRMST热图，用于量化长期绝对差异。")

    add_heading(doc, "3.6 手术入路文献综合", 2)
    add_paragraph(
        doc,
        f"文献综合共纳入{metrics['part2_studies_n']}项研究。三类手术入路家族的合并ELN分布分别为：标准/常规DP {standard_row['combined_summary']}、RAMPS {ramps_row['combined_summary']}、后方/动脉优先谱系 {posterior_row['combined_summary']}。在阈值实现概率层面，标准/常规DP的P(ELN≥12)、P(ELN≥14)和P(ELN≥21)分别为58.4%、47.9%和16.5%；RAMPS分别为74.9%、67.9%和40.0%；后方/动脉优先谱系分别为95.1%、93.2%和82.7%，但该谱系目前仅由极少量小样本研究支撑。",
    )
    add_paragraph(
        doc,
        "从临床解释看，RAMPS相较标准/常规DP可显著提高跨越ELN≥21阈值的概率，是目前最可标准化推广的“升级方案”；后方/动脉优先技术虽显示更高ELN获取趋势，但证据不充分，暂不宜进行正式优效推断。",
    )
    add_figure(doc, PART2_DIR / "figures" / "Figure_4_Approach_ELN_Landscape.png", "图4 不同手术入路家族的已发表队列ELN景观图。散点及误差线显示各队列ELN中心值与离散度，虚线为ELN=12、14和21三个生存锚值。")
    add_figure(doc, PART2_DIR / "figures" / "Figure_5_Station_Accessibility.png", "图5 不同手术入路家族的淋巴结站可及性热图。颜色强度代表基于文献证据支持的可及程度。")
    add_figure(doc, PART2_DIR / "figures" / "Figure_6_RAMPS_MultiStudy_Forest.png", "图6 不同手术入路家族达到ELN≥21的概率森林图。图中显示标准/常规DP与RAMPS的队列级及家族级估计值；后方/动脉优先谱系仅作为低确定性趋势展示。")
    add_figure(doc, PART2_DIR / "figures" / "Figure_7_Threshold_Evidence_Map.png", "图7 各手术入路家族ELN分布与SEER生存锚值的关联图。左侧为家族级ELN分布，右侧为P(ELN≥12/14/21)矩阵。")
    add_paragraph(doc, "表3 不同手术入路家族的文献队列ELN统一化结果与阈值实现概率", size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.15, space_after_pt=3)
    add_dataframe_table(doc, table3, font_size=8.5)
    add_paragraph(doc, "表4 不同手术入路家族的淋巴结站可及性", size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.15, space_after_pt=3)
    add_dataframe_table(doc, table4, font_size=9)
    add_paragraph(doc, "表5 手术入路家族层面的ELN阈值实现概率与临床解释", size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, line_spacing=1.15, space_after_pt=3)
    add_dataframe_table(doc, table5, font_size=8.5)

    add_heading(doc, "4 讨论", 1)
    discussion_paragraphs = [
        "本研究将SEER人群生存分析与手术路径证据综合串联起来，建立了胰腺体尾部PDAC切除术后ELN的双阈值框架：ELN≥12对应最低质量基准，ELN≈21对应高收益目标。前者解决“分期是否充分”的问题，后者更接近“边际收益开始递减、关联减弱”的位置。",
        "ELN=12之所以重要，在于它与现行NCCN最低推荐阈值一致，也与既往大样本SEER研究方向相符[4,8]。在本研究中，ELN≥12组的中位OS由20.0个月提高至26.0个月，且60个月OW-RMST净获益达到2.59个月。这提示足够的送检淋巴结数首先是分期质量控制指标，而不应被简单理解为“多切即多治”的直接治疗效应。",
        "ELN=21则更适合被理解为高收益目标而非严格的生物学临界点。分段Cox变化点和B样条关联减弱趋势均支持该位置的临床意义，但正式Bootstrap区间较宽（6.0～35.0），说明拐点估计存在不可忽视的不确定性。因此，在临床实践中，将21枚视为“希望尽量达到的上位目标”比将其僵化为绝对分界值更为合理。",
        "POS分析为上述解释提供了补充。POS=0至POS≥1的显著风险阶跃以及POS增加后的持续风险累积，提示更充分的ELN并非仅仅增加“数字上的淋巴结总数”，其真正价值在于减少假N0、提高N分期准确性，并使真正N+患者更可能获得更有针对性的辅助治疗[20,29]。",
        "从手术路径角度看，RAMPS在当前证据体系中最符合“标准化升级方案”的定位。与标准/常规DP相比，RAMPS家族的P(ELN≥21)由16.5%提升至40.0%，相对风险约为2.43倍，说明其确有更强的达到高ELN目标的能力。但现有比较研究和最新系统综述/Meta分析均未证实其对总体生存具有明确优效[16,17,23,28]。因此，RAMPS的价值更应被表述为“提高高质量淋巴结评估概率的手术策略”，而非已经证实直接改善OS的手术方式。",
        "后方/动脉优先谱系在本研究中显示出最高的ELN获取趋势，但该结果主要来自极少量小样本研究，且病例选择和术者经验偏倚均较难控制。故本研究将其定位为趋势支持、低确定性的技术方向，而非可直接推广的常规优选路径。",
        "本研究仍存在若干局限。首先，SEER缺乏新辅助治疗细节、术者经验、病理取材规范及真实手术入路等关键信息，ELN与OS之间的正向关联可能部分反映中心整体医疗质量。其次，文献综合基于发表层面的汇总数据，无法进行患者级混杂校正。再次，家族级阈值实现概率依赖正态近似和统一化均值±标准差，无法完全重建偏态分布。最后，手术路径证据主要来自回顾性比较研究，高质量前瞻性验证仍然不足。",
    ]
    for text in discussion_paragraphs:
        add_paragraph(doc, text)

    add_heading(doc, "5 结论", 1)
    add_paragraph(
        doc,
        "在胰腺体尾部PDAC根治性切除术中，ELN≥12可作为淋巴结评估充分性的最低质量基准，ELN≥21可作为追求更高分期收益的临床目标。RAMPS是目前最具文献支撑、能够显著提高达到高ELN目标概率的标准化手术升级方案，但其直接改善总体生存的证据仍不足。对于后方/动脉优先谱系，则需在规范化、多中心研究中进一步验证其真实临床价值。",
    )

    add_heading(doc, "6 声明", 1)
    add_paragraph(doc, "伦理批准与知情同意：本研究使用SEER公开去标识化数据，依据数据使用协议免于机构伦理审批及患者知情同意。", first_indent=False)
    add_paragraph(doc, "利益冲突：作者声明不存在与本文相关的利益冲突。", first_indent=False)
    add_paragraph(doc, "资金来源：____________________", first_indent=False)
    add_paragraph(doc, "数据可及性：SEER原始数据库可公开获取；本研究的分析输出和补充图表已整理于当前目录的相关输出文件夹。", first_indent=False)
    add_paragraph(doc, "作者贡献：____________________", first_indent=False)

    add_heading(doc, "参考文献", 1)
    for ref in refs:
        add_reference(doc, ref)

    doc.save(OUTPUT_DOCX)


if __name__ == "__main__":
    main()
