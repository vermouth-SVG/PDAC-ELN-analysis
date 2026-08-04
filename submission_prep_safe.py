from __future__ import annotations

import copy
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lxml import etree
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


ROOT_DIR = Path.cwd()
OFFICIAL_DIR = ROOT_DIR / "正式投稿内容"

SRC_MANUSCRIPT = OFFICIAL_DIR / "manuscript_patch_safe.docx"
SRC_SUPP_FIG = OFFICIAL_DIR / "supplementary_figures_patch_safe.docx"
SRC_SUPP_TAB = OFFICIAL_DIR / "supplementary_tables_patch_safe.xlsx"

OUT_FIG_RENUMBERED = OFFICIAL_DIR / "revised_manuscript_figrenumbered.docx"
OUT_MANUSCRIPT = OFFICIAL_DIR / "manuscript_submission_polished.docx"
OUT_SUPP_FIG = OFFICIAL_DIR / "supplementary_figures_submission_polished.docx"
OUT_SUPP_TAB_SUB = OFFICIAL_DIR / "supplementary_tables_submission_ready.xlsx"
OUT_SUPP_TAB_ARCH = OFFICIAL_DIR / "supplementary_tables_archive_full.xlsx"
OUT_FIG_LOG = OFFICIAL_DIR / "figure_renumber_log.md"
OUT_CIT_LOG = OFFICIAL_DIR / "citation_conversion_log.md"
OUT_CIT_UNRESOLVED = OFFICIAL_DIR / "unresolved_citations.md"
OUT_NUM_LOG = OFFICIAL_DIR / "number_format_log.md"
OUT_NUM_ISSUES = OFFICIAL_DIR / "number_format_issues.md"
OUT_SUPP_TAB_MAP = OFFICIAL_DIR / "supplementary_table_mapping.md"
OUT_SUPP_TAB_CALLOUTS = OFFICIAL_DIR / "supplement_table_callouts_log.md"
OUT_STROBE = OFFICIAL_DIR / "STROBE_checklist_draft.xlsx"
OUT_STROBE_GAPS = OFFICIAL_DIR / "strobe_gap_summary.md"
OUT_KM_AUDIT = OFFICIAL_DIR / "km_plot_audit.md"
OUT_KM_FIX = OFFICIAL_DIR / "km_plot_fix_log.md"
OUT_KM_UNRESOLVED = OFFICIAL_DIR / "km_plot_unresolved.md"
OUT_FINAL_ISSUES = OFFICIAL_DIR / "final_submission_issues.md"


@dataclass
class TableDecision:
    original_sheet: str
    submission_id: str | None
    new_sheet_name: str | None
    keep_for_submission: bool
    explicit_callout: str
    rationale: str
    clear_title: str


TABLE_DECISIONS = [
    TableDecision(
        "Table1_ELN_Bin_Survival_Overall",
        "ST01",
        "ST01_ELN_Bin_Survival",
        True,
        "结果3.2/3.4（新增 ST01-ST05）",
        "直接支撑 ELN 分箱生存概览，是阈值框架的基础数据。",
        "ELN bins and overall survival in the SEER cohort",
    ),
    TableDecision(
        "Table2_LogRank_Thresholds",
        "ST02",
        "ST02_LogRank_Thresholds",
        True,
        "结果3.2（新增 ST01-ST05）",
        "直接支撑 log-rank 阈值网格搜索结果。",
        "Log-rank threshold grid for ELN cut-points",
    ),
    TableDecision(
        "Table3_ELN_Exact_OS_Curve",
        "ST03",
        "ST03_Exact_OS_Curve",
        True,
        "结果3.2（新增 ST01-ST05）",
        "补充展示精确 ELN 值与中位 OS 的关系，支撑非线性趋势。",
        "Exact ELN values and smoothed median OS",
    ),
    TableDecision(
        "Table4_SEER_ELN_Cox_Model",
        "ST04",
        "ST04_ELN_Cox_Model",
        True,
        "结果3.2/3.3（新增 ST01-ST05）",
        "直接支撑多变量 ELN Cox 模型结果。",
        "Multivariable Cox model for ELN in SEER",
    ),
    TableDecision(
        "Table5_ELN_PH_Test",
        None,
        None,
        False,
        "未在正文单独引用",
        "属于技术性比例风险检验输出，投稿时可归档保留而不进入提交版补充表。",
        "Proportional-hazards test for ELN model",
    ),
    TableDecision(
        "Table6_ELN_Spline_HR_Curve",
        "ST05",
        "ST05_ELN_Spline_HR",
        True,
        "结果3.2（新增 ST01-ST05）",
        "直接支撑 ELN 非线性 HR 曲线。",
        "Spline-derived hazard ratio curve for ELN",
    ),
    TableDecision(
        "Table7_ELN_DCA_12m",
        None,
        None,
        False,
        "未在正文单独引用",
        "属于技术性 DCA 输出，本轮主文未围绕 DCA 展开，归档保留即可。",
        "Decision-curve analysis for ELN at 12 months",
    ),
    TableDecision(
        "Table8_ELN_DCA_36m",
        None,
        None,
        False,
        "未在正文单独引用",
        "属于技术性 DCA 输出，本轮主文未围绕 DCA 展开，归档保留即可。",
        "Decision-curve analysis for ELN at 36 months",
    ),
    TableDecision(
        "Table9_ELN_DCA_60m",
        None,
        None,
        False,
        "未在正文单独引用",
        "属于技术性 DCA 输出，本轮主文未围绕 DCA 展开，归档保留即可。",
        "Decision-curve analysis for ELN at 60 months",
    ),
    TableDecision(
        "Table10_SEER_POS_Cox_Model",
        "ST06",
        "ST06_POS_Cox_Model",
        True,
        "结果3.5（新增 ST06-ST08）",
        "直接支撑 POS 阶跃与 Cox 结果。",
        "Multivariable Cox model for POS in SEER",
    ),
    TableDecision(
        "Table11_POS_PH_Test",
        None,
        None,
        False,
        "未在正文单独引用",
        "技术性比例风险检验输出，投稿时归档保留即可。",
        "Proportional-hazards test for POS model",
    ),
    TableDecision(
        "Table12_POS_Spline_HR_Curve",
        "ST07",
        "ST07_POS_Spline_HR",
        True,
        "结果3.5（新增 ST06-ST08）",
        "直接支撑 POS 连续样条风险曲线。",
        "Spline-derived hazard ratio curve for POS",
    ),
    TableDecision(
        "Table13_POS_Delta_RMST",
        "ST08",
        "ST08_POS_Delta_RMST",
        True,
        "结果3.5（新增 ST06-ST08）",
        "直接支撑 POS 分层 RMST 差异。",
        "Delta RMST across POS strata",
    ),
    TableDecision(
        "Table14_Study_Catalog",
        "ST09",
        "ST09_Study_Catalog",
        True,
        "结果3.6（新增 ST09-ST13）",
        "作为文献综合纳入研究目录，直接支撑文献证据来源。",
        "Study catalog for the surgical-approach evidence synthesis",
    ),
    TableDecision(
        "Table15_Approach_Comparison",
        "ST10",
        "ST10_Approach_Comparison",
        True,
        "结果3.6（新增 ST09-ST13）",
        "直接支撑不同术式家族的阈值实现概率比较。",
        "Approach-family comparison across ELN thresholds",
    ),
    TableDecision(
        "Table16_Anatomical_Access",
        "ST11",
        "ST11_Anatomical_Access",
        True,
        "结果3.6（新增 ST09-ST13）",
        "作为术式解剖学可及性说明，支撑 RAMPS 的定位。",
        "Anatomical access profile of each approach family",
    ),
    TableDecision(
        "Table17_Statistical_Validation",
        "ST12",
        "ST12_Stat_Validation",
        True,
        "结果3.6（新增 ST09-ST13）",
        "支撑家族级阈值概率差异的统计比较。",
        "Validation of threshold-probability comparisons",
    ),
    TableDecision(
        "Table18_Anchor_Attainment",
        "ST13",
        "ST13_Anchor_Attainment",
        True,
        "结果3.6（新增 ST09-ST13）",
        "作为 ELN 锚值实现概率与推荐定位的总览表，投稿价值高。",
        "Anchor-attainment summary across approach families",
    ),
]


def para_text(p) -> str:
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def nonempty_body_paragraphs(root):
    paras = []
    for p in root.xpath(".//w:body/w:p", namespaces=NS):
        text = para_text(p)
        if text.strip():
            paras.append(p)
    return paras


def ensure_rpr(run):
    rpr = run.find(f"{{{W_NS}}}rPr")
    if rpr is None:
        rpr = etree.Element(f"{{{W_NS}}}rPr")
        run.insert(0, rpr)
    return rpr


def build_run(text: str, template_run, superscript: bool = False):
    run = etree.Element(f"{{{W_NS}}}r")
    if template_run is not None:
        template_rpr = template_run.find(f"{{{W_NS}}}rPr")
        if template_rpr is not None:
            run.append(copy.deepcopy(template_rpr))
    if superscript:
        rpr = ensure_rpr(run)
        for existing in rpr.findall(f"{{{W_NS}}}vertAlign"):
            rpr.remove(existing)
        vert = etree.SubElement(rpr, f"{{{W_NS}}}vertAlign")
        vert.set(f"{{{W_NS}}}val", "superscript")
    else:
        rpr = run.find(f"{{{W_NS}}}rPr")
        if rpr is not None:
            for existing in rpr.findall(f"{{{W_NS}}}vertAlign"):
                rpr.remove(existing)
    text_node = etree.SubElement(run, f"{{{W_NS}}}t")
    if text.startswith(" ") or text.endswith(" "):
        text_node.set(XML_SPACE, "preserve")
    text_node.text = text
    return run


def rebuild_paragraph(p, segments: Iterable[tuple[str, bool]]):
    template_run = p.find(f"{{{W_NS}}}r")
    ppr = p.find(f"{{{W_NS}}}pPr")
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    for text, superscript in segments:
        if not text:
            continue
        p.append(build_run(text, template_run, superscript))


def parse_docx_xml(docx_path: Path):
    with zipfile.ZipFile(docx_path, "r") as zf:
        return etree.fromstring(zf.read("word/document.xml"))


def save_docx_with_xml(src_docx: Path, dst_docx: Path, xml_bytes: bytes):
    with zipfile.ZipFile(src_docx, "r") as zin, zipfile.ZipFile(dst_docx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = xml_bytes if info.filename == "word/document.xml" else zin.read(info.filename)
            zout.writestr(info, data)


def find_reference_header_index(paras) -> int:
    for idx, p in enumerate(paras, start=1):
        if para_text(p).strip() == "参考文献":
            return idx
    raise RuntimeError("Could not locate the references header.")


def replace_thousands(text: str) -> str:
    pattern = re.compile(r"(?<!\d)(\d{1,3}),(\d{3})(?!\d)")
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r"\1\2", text)
    return text


def convert_citation_segments(text: str):
    pattern = re.compile(r"(\[\d+(?:[-,]\d+)*\])")
    parts = pattern.split(text)
    segments = []
    conversions = []
    for part in parts:
        if not part:
            continue
        if pattern.fullmatch(part):
            cleaned = part[1:-1]
            segments.append((cleaned, True))
            conversions.append((part, cleaned))
        else:
            segments.append((part, False))
    return segments, conversions


def render_md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_text(path: Path, text: str):
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_figure_renumber_doc():
    root = parse_docx_xml(SRC_MANUSCRIPT)
    paras = nonempty_body_paragraphs(root)
    ref_idx = find_reference_header_index(paras)

    figure_logs = []
    pattern = re.compile(r"图([246])(?=[^0-9]|$)")
    mapping = {"2": "1", "4": "2", "6": "3"}

    for idx, p in enumerate(paras[: ref_idx - 1], start=1):
        original = para_text(p)
        updated = pattern.sub(lambda m: f"图{mapping[m.group(1)]}", original)
        if updated != original:
            rebuild_paragraph(p, [(updated, False)])
            figure_logs.append((idx, original, updated))

    xml_bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
    save_docx_with_xml(SRC_MANUSCRIPT, OUT_FIG_RENUMBERED, xml_bytes)
    return figure_logs


def build_polished_manuscript():
    root = parse_docx_xml(SRC_MANUSCRIPT)
    paras = nonempty_body_paragraphs(root)
    ref_idx = find_reference_header_index(paras)

    figure_logs = []
    citation_logs = []
    unresolved_citations = []
    number_logs = []
    callout_logs = []

    figure_pattern = re.compile(r"图([246])(?=[^0-9]|$)")
    figure_mapping = {"2": "1", "4": "2", "6": "3"}
    citation_para_numbers = {8, 9, 10, 11, 52, 54, 55}
    callouts = {
        34: "相关补充表见ST01-ST05。",
        44: "相关补充表见ST06-ST08。",
        47: "相关补充表见ST09-ST13。",
    }

    for idx, p in enumerate(paras, start=1):
        original = para_text(p)

        if idx < ref_idx:
            renumbered = figure_pattern.sub(lambda m: f"图{figure_mapping[m.group(1)]}", original)
            if renumbered != original:
                rebuild_paragraph(p, [(renumbered, False)])
                figure_logs.append((idx, original, renumbered))
                original = renumbered

            text = para_text(p)
            if idx in citation_para_numbers:
                segments, conversions = convert_citation_segments(text)
                if conversions:
                    rebuild_paragraph(p, segments)
                    citation_logs.append((idx, conversions))
                text = para_text(p)
                if "[" in text and "]" in text:
                    unresolved_citations.append((idx, text))

            text = para_text(p)
            no_commas = replace_thousands(text)
            if no_commas != text:
                rebuild_paragraph(p, [(no_commas, False)])
                number_logs.append((f"段落 {idx}", text, no_commas))
                text = no_commas

            if idx in callouts and callouts[idx] not in text:
                updated = text + callouts[idx]
                rebuild_paragraph(p, [(updated, False)])
                callout_logs.append((idx, callouts[idx]))

    for table_index, tbl in enumerate(root.xpath(".//w:tbl", namespaces=NS), start=1):
        for row_index, row in enumerate(tbl.xpath("./w:tr", namespaces=NS), start=1):
            for col_index, cell in enumerate(row.xpath("./w:tc", namespaces=NS), start=1):
                cell_text = "".join(cell.xpath(".//w:t/text()", namespaces=NS))
                updated = replace_thousands(cell_text)
                if updated == cell_text:
                    continue
                for p in cell.xpath(".//w:p", namespaces=NS):
                    if para_text(p) == cell_text:
                        rebuild_paragraph(p, [(updated, False)])
                        number_logs.append((f"表{table_index} R{row_index}C{col_index}", cell_text, updated))
                        break

    xml_bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
    save_docx_with_xml(SRC_MANUSCRIPT, OUT_MANUSCRIPT, xml_bytes)
    return figure_logs, citation_logs, unresolved_citations, number_logs, callout_logs


def build_submission_workbooks():
    shutil.copy2(SRC_SUPP_TAB, OUT_SUPP_TAB_ARCH)

    wb = load_workbook(SRC_SUPP_TAB)
    original_order = wb.sheetnames[:]

    for decision in TABLE_DECISIONS:
        if not decision.keep_for_submission:
            del wb[decision.original_sheet]

    contents = wb.create_sheet("Contents_Legends", 0)
    headers = ["ST ID", "Submission sheet", "Original sheet", "Clear title", "Main-text callout", "Rationale"]
    for col, header in enumerate(headers, start=1):
        cell = contents.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)

    row_idx = 2
    for decision in TABLE_DECISIONS:
        if not decision.keep_for_submission:
            continue
        ws = wb[decision.original_sheet]
        ws.title = decision.new_sheet_name
        contents.cell(row=row_idx, column=1, value=decision.submission_id)
        contents.cell(row=row_idx, column=2, value=decision.new_sheet_name)
        contents.cell(row=row_idx, column=3, value=decision.original_sheet)
        contents.cell(row=row_idx, column=4, value=decision.clear_title)
        contents.cell(row=row_idx, column=5, value=decision.explicit_callout)
        contents.cell(row=row_idx, column=6, value=decision.rationale)
        row_idx += 1

    contents.freeze_panes = "A2"
    contents.column_dimensions["A"].width = 10
    contents.column_dimensions["B"].width = 28
    contents.column_dimensions["C"].width = 34
    contents.column_dimensions["D"].width = 48
    contents.column_dimensions["E"].width = 24
    contents.column_dimensions["F"].width = 48

    # Preserve kept-sheet order after renaming.
    target_order = ["Contents_Legends"] + [d.new_sheet_name for d in TABLE_DECISIONS if d.keep_for_submission]
    wb._sheets = [wb[name] for name in target_order]
    wb.save(OUT_SUPP_TAB_SUB)

    mapping_rows = []
    for sheet_name in original_order:
        decision = next(d for d in TABLE_DECISIONS if d.original_sheet == sheet_name)
        mapping_rows.append(
            [
                sheet_name,
                decision.submission_id or "-",
                decision.new_sheet_name or "-",
                "保留" if decision.keep_for_submission else "归档",
                decision.explicit_callout,
                decision.rationale,
            ]
        )
    return mapping_rows


def build_strobe_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "STROBE"
    headers = ["Item", "STROBE item", "Current manuscript location", "Status", "Minimal next-step suggestion"]
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)

    rows = [
        ("1", "Title / abstract", "标题；摘要第3-6段", "Partial", "在英文稿阶段把摘要改成更标准的 Introduction/Methods/Results/Conclusion 标识，并在摘要中明确 retrospective cohort design。"),
        ("2", "Background / rationale", "引言第8-12段", "Yes", "保持当前框架即可。"),
        ("3", "Objectives", "引言第12段", "Yes", "保持当前双问题目标表述。"),
        ("4", "Study design", "方法第15-25段", "Partial", "在方法开头补一处“基于 SEER 的回顾性观察性队列研究”即可。"),
        ("5", "Setting", "方法第15段、第23段", "Partial", "后续可补充纳入年份、数据库版本和文献检索时间点。"),
        ("6", "Participants", "方法第15段；补充图S1", "Partial", "可再明确纳入/排除标准的简短列表。"),
        ("7", "Variables", "方法第17-21段", "Yes", "保持当前变量定义。"),
        ("8", "Data sources / measurement", "方法第15-25段", "Partial", "可补一处对 SEER 变量抽取与术式文献指标统一化公式的简述。"),
        ("9", "Bias", "讨论第57段（局限）", "Partial", "在方法或讨论中单句注明选择偏倚、残余混杂和病理取材差异。"),
        ("10", "Study size", "方法第15段；结果第28段", "Yes", "保持当前样本量报告。"),
        ("11", "Quantitative variables", "方法第17-25段", "Yes", "保持当前阈值、样条、RMST 描述。"),
        ("12", "Statistical methods", "方法第17-25段", "Partial", "可补缺失值处理与敏感性分析组织方式的更明确描述。"),
        ("13", "Participants (results)", "结果第28段；补充图S1", "Yes", "保持当前报告。"),
        ("14", "Descriptive data", "结果第28-31段；表1", "Yes", "保持当前表1。"),
        ("15", "Outcome data", "结果第33-47段", "Yes", "保持当前 OS、HR、RMST、POS、术式综合结果。"),
        ("16", "Main results", "结果第33-47段", "Yes", "保持当前主结果。"),
        ("17", "Other analyses", "结果第40-47段；补充图S4-S8；补充表ST01-ST13", "Yes", "保持当前补充分析层级。"),
        ("18", "Key results", "讨论第51-56段", "Yes", "保持当前主线总结。"),
        ("19", "Limitations", "讨论第57段", "Yes", "保持当前局限段落。"),
        ("20", "Interpretation", "讨论第51-56段", "Yes", "保持当前解释框架。"),
        ("21", "Generalisability", "讨论第55-57段", "Partial", "英文稿阶段可更明确区分 SEER 外部适用性与术式文献异质性。"),
        ("22", "Funding", "声明第61段（占位）", "Partial", "需人工补全真实 funding / no funding 信息。"),
    ]

    for row_idx, row in enumerate(rows, start=2):
        for col_idx, value in enumerate(row, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)

    widths = {"A": 10, "B": 22, "C": 34, "D": 12, "E": 54}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"
    wb.save(OUT_STROBE)


def build_markdown_outputs(
    figure_logs,
    citation_logs,
    unresolved_citations,
    number_logs,
    callout_logs,
    table_mapping_rows,
):
    figure_lines = [
        "# Figure Renumber Log",
        "",
        "- 主文图号连续化映射：图2 → 图1；图4 → 图2；图6 → 图3。",
        "",
        render_md_table(
            ["段落", "修改前", "修改后"],
            [[str(idx), old, new] for idx, old, new in figure_logs],
        ),
    ]
    write_text(OUT_FIG_LOG, "\n".join(figure_lines))

    citation_rows = []
    for idx, conversions in citation_logs:
        citation_rows.append(
            [
                str(idx),
                "；".join(old for old, _ in conversions),
                "；".join(new for _, new in conversions),
            ]
        )
    cit_lines = [
        "# Citation Conversion Log",
        "",
        "- 已按 BJS Vancouver 文内引文要求，将正文命中的方括号引用转换为 Word superscript run。",
        "",
        render_md_table(["段落", "原始形式", "转换后（上标内容）"], citation_rows),
    ]
    write_text(OUT_CIT_LOG, "\n".join(cit_lines))

    if unresolved_citations:
        unresolved_lines = [
            "# Unresolved Citations",
            "",
            render_md_table(
                ["段落", "残留文本"],
                [[str(idx), text] for idx, text in unresolved_citations],
            ),
        ]
    else:
        unresolved_lines = [
            "# Unresolved Citations",
            "",
            "- 未发现需要人工继续处理的正文残留方括号引文。",
        ]
    write_text(OUT_CIT_UNRESOLVED, "\n".join(unresolved_lines))

    num_lines = [
        "# Number Format Log",
        "",
        "- 按用户要求，仅在正文、摘要和表头的安全位置去掉千位逗号；未改动小数点、95% CI、P 值等格式。",
        "",
        render_md_table(["位置", "修改前", "修改后"], [[loc, old, new] for loc, old, new in number_logs]),
    ]
    write_text(OUT_NUM_LOG, "\n".join(num_lines))

    write_text(
        OUT_NUM_ISSUES,
        "\n".join(
            [
                "# Number Format Issues",
                "",
                "- 本轮未发现正文或表头中残留且需要继续人工清理的千位逗号；参考文献列表中的 22,045 未处理，因本轮明确不改参考文献条目内容。",
                "- 需要注意：BJS 官方习惯是“千位后使用空格而不是逗号”（例如 10 000）。本轮按用户要求采用了更安全的去逗号方案（例如 5107），以避免 Word 排版风险；若进入英文投稿定稿阶段，建议再做一次全稿人工统一。",
            ]
        ),
    )

    map_lines = [
        "# Supplementary Table Mapping",
        "",
        "- 投稿版补充表采用 `ST01-ST13` 体系；未入选的技术性工作表保留在 `supplementary_tables_archive_full.xlsx`。",
        "",
        render_md_table(
            ["原始 sheet", "投稿编号", "投稿 sheet", "处理", "正文引用", "理由"],
            table_mapping_rows,
        ),
    ]
    write_text(OUT_SUPP_TAB_MAP, "\n".join(map_lines))

    if callout_logs:
        rows = [[str(idx), text] for idx, text in callout_logs]
        callout_lines = [
            "# Supplement Table Callouts Log",
            "",
            "- 为满足 BJS“补充材料需在正文中引用”的要求，仅在结果部分补了 3 处短句式补充表挂接。",
            "",
            render_md_table(["段落", "新增短句"], rows),
        ]
    else:
        callout_lines = [
            "# Supplement Table Callouts Log",
            "",
            "- 本轮未新增正文补充表挂接。",
        ]
    write_text(OUT_SUPP_TAB_CALLOUTS, "\n".join(callout_lines))

    gap_lines = [
        "# STROBE Gap Summary",
        "",
        "1. 研究设计类型需要更明确：当前方法部分已有观察性研究内容，但建议在英文稿中直接写明 retrospective cohort study。",
        "2. 偏倚控制写得还不够显式：选择偏倚、残余混杂、病理取材差异可在方法或局限中再明确一句。",
        "3. 数据来源与测量统一化步骤可再压缩补充：尤其是术式文献均值±SD 折算规则。",
        "4. Funding / CRediT 仍是占位：这属于必须人工补全的信息，不建议机器代填。",
        "5. 外部适用性可进一步单独表述：区分 SEER 人群结论与文献综合推断的推广边界。",
    ]
    write_text(OUT_STROBE_GAPS, "\n".join(gap_lines))

    km_lines = [
        "# KM Plot Audit",
        "",
        "- 审核范围：主文重编号后图1-3，以及补充图 S2、S4、S6。",
        "- 由于当前文件夹仅有 docx 内嵌位图，且无原始绘图脚本/可编辑图源，本轮只能做题注级识别和人工复核提示，不能可靠重画或自动补 at-risk table。",
        "",
        render_md_table(
            ["图", "是否 KM", "number-at-risk table", "坐标轴/图例", "尾部截断", "结论"],
            [
                ["主文图1（原图2）", "否", "N/A", "N/A", "N/A", "非 KM 图，无需处理。"],
                ["主文图2（原图4）", "否", "N/A", "N/A", "N/A", "非 KM 图，无需处理。"],
                ["主文图3（原图6）", "否", "N/A", "N/A", "N/A", "非 KM 图，无需处理。"],
                ["补充图 S2", "是（题注明确含 KM）", "无法仅凭 docx 位图机读确认", "需人工目视检查", "需人工目视检查", "建议人工确认是否附 at-risk table、是否在尾部低风险人数时截断。"],
                ["补充图 S4", "不确定（题注仅示 stage 分层）", "无法判定", "需人工目视检查", "需人工目视检查", "需打开源图人工确认是否含 KM 面板。"],
                ["补充图 S6", "是（题注明确含 KM）", "无法仅凭 docx 位图机读确认", "需人工目视检查", "需人工目视检查", "建议人工确认是否附 at-risk table、是否在尾部低风险人数时截断。"],
            ],
        ),
    ]
    write_text(OUT_KM_AUDIT, "\n".join(km_lines))

    write_text(
        OUT_KM_FIX,
        "\n".join(
            [
                "# KM Plot Fix Log",
                "",
                "- 未执行自动修图。",
                "- 原因：当前目录无可编辑绘图脚本或单独源图文件；依据用户限制，不对结果图进行擅自重画。",
            ]
        ),
    )

    write_text(
        OUT_KM_UNRESOLVED,
        "\n".join(
            [
                "# KM Plot Unresolved",
                "",
                "1. 补充图 S2：需人工确认 panel B 是否附 number-at-risk table。",
                "2. 补充图 S4：需人工确认是否包含 KM 曲线；若包含，也需确认 at-risk table 与尾部截断。",
                "3. 补充图 S6：需人工确认 panel C 是否附 number-at-risk table。",
            ]
        ),
    )

    final_lines = [
        "# Final Submission Issues",
        "",
        "本轮整理参照 BJS 官方投稿说明（Oxford Academic，访问日期：2026-04-19）：Original article 一般限 3000 词、主文图表合计 5 个、摘要需 <250 词结构化、文内引用用 Vancouver superscript、补充材料须在正文中引用，KM 曲线需附 number-at-risk table，并在人数过少时截断。",
        "",
        "## 已修好的问题",
        "",
        "1. 主文主图已连续化为图1-图3，旧的图2/图4/图6 编号已统一替换。",
        "2. 正文命中的方括号引用已转换为 superscript 数字引用，未重排参考文献编号。",
        "3. 正文和表1表头中的千位逗号已按最保守方式去掉，未动小数、95% CI 或 P 值格式。",
        "4. 补充图已确认保持 S1-S8，不再改编号，并输出为新的提交版文件。",
        "5. 补充表已拆分为投稿版与归档版；投稿版只保留与主文论证直接相关的 13 张表，并新增 contents/legends sheet。",
        "6. 正文仅新增 3 处短句式补充表挂接，以满足 BJS 对 supplementary material 的引用要求。",
        "7. 已生成 STROBE checklist 草稿与 gap summary，可直接作为后续人工完善基础。",
        "",
        "## 仍未修好的问题",
        "",
        "1. KM 曲线是否带 number-at-risk table 目前无法从 docx 内嵌位图安全自动确认；已输出审计清单，但未重画。",
        "2. 主文仍为中文稿，不能直接用于 BJS 正式投稿。",
        "3. 参考文献列表条目内容本轮未全面按 Vancouver 成品格式重修，只改了正文引用样式。",
        "4. 当前数字格式已去掉千位逗号，但尚未统一成 BJS 官方偏好的“千位空格”写法；这是后续英文定稿时应人工统一的格式项。",
        "5. 补充表当前采用 `ST01-ST13` 体系以服务中文整理；BJS 最终上传时仍需人工确认是否改成 `Table S1-S13`。",
        "6. 主文尚未补入 figure alt text，而 BJS 要求在主文图注下直接提供 `Alt text:` 描述。",
        "",
        "## 必须人工处理的事项",
        "",
        "1. title page：作者全名、单位、通讯作者、funding、conflict、data availability、投稿类别。",
        "2. CRediT author contributions 与真实 funding/no funding 声明。",
        "3. 若进入正式提交流程，需要把主图导出为单独 `Figure_1`、`Figure_2`、`Figure_3` 图像文件；BJS 不接受把最终图文件作为 Word figure file 上传。",
        "4. 人工核查补充图 S2/S4/S6 的 KM 面板是否带 at-risk table、坐标轴是否清晰、尾部是否应截断。",
        "5. 人工确认投稿时是否保留当前中文补充表编号体系，或统一改成 BJS 常见 `Table Sx` 体系。",
        "",
        "## 建议放到英文稿阶段再处理",
        "",
        "1. 全文英文翻译与投稿式语言压缩。",
        "2. 摘要标题词改成更贴近 BJS 的 Introduction/Methods/Results/Conclusion。",
        "3. 参考文献列表的英文期刊缩写、卷期页码、DOI 全面规范化。",
        "4. 主图和图注的英文命名、alt text、可访问性检查。",
        "",
        "## 当前阶段判断",
        "",
        "1. 继续中文定稿：可以。",
        "2. 进入英文翻译：可以。",
        "3. 直接投稿：不能直接投稿。",
        "",
        "BJS 官方说明链接：",
        "https://academic.oup.com/bjs/pages/general-instructions",
    ]
    write_text(OUT_FINAL_ISSUES, "\n".join(final_lines))


def validate_outputs():
    root = parse_docx_xml(OUT_MANUSCRIPT)
    paras = nonempty_body_paragraphs(root)
    ref_idx = find_reference_header_index(paras)
    body_text = "\n".join(para_text(p) for p in paras[: ref_idx - 1])

    problems = []
    required_tokens = ["POS≥1", "95% CI"]
    for token in required_tokens:
        if token not in body_text:
            problems.append(f"Missing preserved token: {token}")
    for bad in ["????", "POS?1", "?RMST"]:
        if bad in body_text:
            problems.append(f"Corruption token present: {bad}")
    for old_fig in ["图4 不同手术入路家族的已发表队列ELN景观图", "图6 不同手术入路家族达到ELN≥21的概率森林图"]:
        if old_fig in body_text:
            problems.append(f"Residual old figure caption: {old_fig}")

    bracket_pattern = re.compile(r"\[\d+(?:[-,]\d+)*\]")
    for idx in [8, 9, 10, 11, 52, 54, 55]:
        text = para_text(paras[idx - 1])
        if bracket_pattern.search(text):
            problems.append(f"Residual bracket citation in paragraph {idx}: {text}")

    return problems


def verify_supp_fig_and_copy():
    root = parse_docx_xml(SRC_SUPP_FIG)
    paras = nonempty_body_paragraphs(root)
    expected = [f"Supplementary Figure S{i}." for i in range(1, 9)]
    actual = [para_text(p).strip() for p in paras[:8]]
    for expected_prefix, actual_text in zip(expected, actual, strict=True):
        if not actual_text.startswith(expected_prefix):
            raise RuntimeError(f"Supplementary figure numbering mismatch: expected prefix {expected_prefix}, got {actual_text}")
    shutil.copy2(SRC_SUPP_FIG, OUT_SUPP_FIG)


def main():
    figure_logs_only = build_figure_renumber_doc()
    figure_logs, citation_logs, unresolved_citations, number_logs, callout_logs = build_polished_manuscript()
    table_mapping_rows = build_submission_workbooks()
    build_strobe_workbook()
    verify_supp_fig_and_copy()
    build_markdown_outputs(
        figure_logs=figure_logs,
        citation_logs=citation_logs,
        unresolved_citations=unresolved_citations,
        number_logs=number_logs,
        callout_logs=callout_logs,
        table_mapping_rows=table_mapping_rows,
    )
    validation_problems = validate_outputs()
    if validation_problems:
        raise RuntimeError("Validation failed:\n" + "\n".join(validation_problems))

    print("Created:")
    for path in [
        OUT_FIG_RENUMBERED,
        OUT_MANUSCRIPT,
        OUT_SUPP_FIG,
        OUT_SUPP_TAB_SUB,
        OUT_SUPP_TAB_ARCH,
        OUT_FIG_LOG,
        OUT_CIT_LOG,
        OUT_CIT_UNRESOLVED,
        OUT_NUM_LOG,
        OUT_NUM_ISSUES,
        OUT_SUPP_TAB_MAP,
        OUT_SUPP_TAB_CALLOUTS,
        OUT_STROBE,
        OUT_STROBE_GAPS,
        OUT_KM_AUDIT,
        OUT_KM_FIX,
        OUT_KM_UNRESOLVED,
        OUT_FINAL_ISSUES,
    ]:
        print(path.name)
    print(f"Figure renumber changes: {len(figure_logs_only)}")


if __name__ == "__main__":
    main()
