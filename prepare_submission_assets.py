from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from part1_seer_pipeline import (
    AnalysisConfig,
    PART1_MAIN_FIGURES,
    PART1_MAIN_TABLE_SHEETS,
    PART1_SHEET_NAME_MAP,
    PART1_SUPPLEMENTARY_FIGURES,
    build_manuscript_descriptive_table,
    load_and_prepare_dataset,
)
from part2_anatomical_evidence_pipeline import (
    PART2_MAIN_FIGURES,
    PART2_MAIN_TABLE_SHEETS,
    PART2_SHEET_NAME_MAP,
    PART2_SUPPLEMENTARY_FIGURES,
)


ROOT = Path(".")
PART1_DIR = ROOT / "outputs_part1_seer_analysis"
PART2_DIR = ROOT / "outputs_part2_anatomical_evidence"
COMBINED_MAIN_TABLES = ROOT / "Main_Text_Tables_Combined.xlsx"

PART1_RENAME_MAP = {
    "Figure_1_SEER_ELN_Overview": "Figure_1_SEER_ELN_Overview",
    "Figure_2_SEER_ELN_Modeling": "Figure_2_SEER_ELN_Modeling",
    "Figure_4_SEER_POS_Analysis": "Figure_3_SEER_POS_Analysis",
    "Figure_3_SEER_Model_Performance": "Supplementary_Figure_S1_SEER_Model_Performance",
    "Supplementary_Figure_S1_SEER_ELN_By_Stage": "Supplementary_Figure_S2_SEER_ELN_By_Stage",
}

PART2_RENAME_MAP = {
    "Figure_5_Approach_ELN_Landscape": "Figure_4_Approach_ELN_Landscape",
    "Figure_6_Station_Accessibility": "Figure_5_Station_Accessibility",
    "Figure_8_RAMPS_MultiStudy_Forest": "Figure_6_RAMPS_MultiStudy_Forest",
    "Figure_9_Threshold_Evidence_Map": "Figure_7_Threshold_Evidence_Map",
    "Figure_7_Approach_Profile": "Supplementary_Figure_S4_Approach_Profile",
    "Supplementary_Figure_S3_Approach_Profile": "Supplementary_Figure_S4_Approach_Profile",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_sheet_name(name: str) -> str:
    for token in ["[", "]", ":", "*", "?", "/", "\\"]:
        name = name.replace(token, "_")
    return name[:31]


def convert_png_to_pdf_tiff(png_path: Path, dpi: int = 300) -> None:
    image = plt.imread(png_path)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    stem = png_path.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def ensure_multiformat_figures(figure_dir: Path) -> None:
    for png_path in figure_dir.glob("*.png"):
        convert_png_to_pdf_tiff(png_path)


def rename_figure_files(figure_dir: Path, rename_map: dict[str, str]) -> None:
    for old_stem, new_stem in rename_map.items():
        for suffix in [".png", ".pdf", ".tiff"]:
            old_path = figure_dir / f"{old_stem}{suffix}"
            new_path = figure_dir / f"{new_stem}{suffix}"
            if old_path.exists() and old_path != new_path:
                if new_path.exists():
                    new_path.unlink()
                old_path.replace(new_path)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_figure_bundle(figure_dir: Path, stems: list[str], target_dir: Path) -> None:
    reset_dir(target_dir)
    for stem in stems:
        for suffix in [".png", ".pdf", ".tiff"]:
            src = figure_dir / f"{stem}{suffix}"
            if src.exists():
                shutil.copy2(src, target_dir / src.name)


def read_best_cut(workbook_path: Path) -> int:
    xl = pd.ExcelFile(workbook_path)
    summary_sheet = "Study_Summary" if "Study_Summary" in xl.sheet_names else "Summary"
    summary = pd.read_excel(workbook_path, sheet_name=summary_sheet)
    mapping = dict(zip(summary.iloc[:, 0], summary.iloc[:, 1]))
    return int(mapping.get("Best log-rank cut", 12))


def build_part1_descriptive_stats() -> pd.DataFrame:
    config = AnalysisConfig()
    analysis_df, _ = load_and_prepare_dataset(config)
    best_cut = read_best_cut(PART1_DIR / "tables" / "Part1_SEER_Tables.xlsx")
    return build_manuscript_descriptive_table(analysis_df, best_cut)


def load_workbook_data(path: Path) -> list[tuple[str, pd.DataFrame]]:
    xl = pd.ExcelFile(path)
    return [(sheet, pd.read_excel(path, sheet_name=sheet)) for sheet in xl.sheet_names]


def rewrite_workbook(path: Path, mapping: dict[str, str], extra_sheets: list[tuple[str, pd.DataFrame]] | None = None) -> Path:
    sheets = load_workbook_data(path)
    temp_path = path.with_name(f"{path.stem}__tmp{path.suffix}")
    with pd.ExcelWriter(temp_path) as writer:
        if extra_sheets:
            for sheet_name, df in extra_sheets:
                public_name = mapping.get(sheet_name, sheet_name)
                df.to_excel(writer, sheet_name=sanitize_sheet_name(public_name), index=False)
        extra_names = {name for name, _ in extra_sheets} if extra_sheets else set()
        for old_name, df in sheets:
            if old_name in extra_names:
                continue
            public_name = mapping.get(old_name, old_name)
            df.to_excel(writer, sheet_name=sanitize_sheet_name(public_name), index=False)
    final_path = path
    try:
        if path.exists():
            path.unlink()
        temp_path.replace(path)
    except PermissionError:
        final_path = path.with_name(f"{path.stem}_Submission{path.suffix}")
        if final_path.exists():
            final_path.unlink()
        temp_path.replace(final_path)
    return final_path


def write_subset_workbook(src_path: Path, dest_path: Path, keep_sheets: list[str]) -> None:
    ensure_dir(dest_path.parent)
    xl = pd.ExcelFile(src_path)
    with pd.ExcelWriter(dest_path) as writer:
        for sheet in keep_sheets:
            if sheet in xl.sheet_names:
                pd.read_excel(src_path, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)


def write_asset_guide(path: Path, title: str, main_figures: list[str], supp_figures: list[str], main_tables: list[str]) -> None:
    lines = [
        title,
        "",
        "Recommended Main-Text Figures:",
        *[f"- {item}" for item in main_figures],
        "",
        "Recommended Supplementary Figures:",
        *[f"- {item}" for item in supp_figures],
        "",
        "Recommended Main-Text Tables:",
        *[f"- {item}" for item in main_tables],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_combined_main_tables() -> None:
    part1_path = PART1_DIR / "main_text_tables" / "Part1_Main_Text_Tables.xlsx"
    part2_path = PART2_DIR / "main_text_tables" / "Part2_Main_Text_Tables.xlsx"
    part1_order = [
        "Table1_Descriptive_Stats",
        "Table2_SEER_RMST",
    ]
    part2_order = PART2_MAIN_TABLE_SHEETS
    temp_target = COMBINED_MAIN_TABLES.with_name(f"{COMBINED_MAIN_TABLES.stem}__tmp{COMBINED_MAIN_TABLES.suffix}")
    with pd.ExcelWriter(temp_target) as writer:
        for sheet in part1_order:
            pd.read_excel(part1_path, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)
        for sheet in part2_order:
            pd.read_excel(part2_path, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)
    if COMBINED_MAIN_TABLES.exists():
        COMBINED_MAIN_TABLES.unlink()
    temp_target.replace(COMBINED_MAIN_TABLES)
    duplicate = COMBINED_MAIN_TABLES.with_name(f"{COMBINED_MAIN_TABLES.stem}_Submission{COMBINED_MAIN_TABLES.suffix}")
    if duplicate.exists():
        duplicate.unlink()


def package_part1() -> None:
    figure_dir = PART1_DIR / "figures"
    ensure_multiformat_figures(figure_dir)
    rename_figure_files(figure_dir, PART1_RENAME_MAP)
    ensure_multiformat_figures(figure_dir)

    workbook_path = PART1_DIR / "tables" / "Part1_SEER_Tables.xlsx"
    descriptive_df = build_part1_descriptive_stats()
    workbook_for_use = rewrite_workbook(workbook_path, PART1_SHEET_NAME_MAP, extra_sheets=[("Descriptive_Stats", descriptive_df)])

    copy_figure_bundle(figure_dir, PART1_MAIN_FIGURES, PART1_DIR / "main_text_figures")
    copy_figure_bundle(figure_dir, PART1_SUPPLEMENTARY_FIGURES, PART1_DIR / "supplementary_figures")

    write_subset_workbook(workbook_for_use, PART1_DIR / "main_text_tables" / "Part1_Main_Text_Tables.xlsx", PART1_MAIN_TABLE_SHEETS)
    supp_tables_dir = PART1_DIR / "supplementary_tables"
    if supp_tables_dir.exists():
        shutil.rmtree(supp_tables_dir)

    write_asset_guide(
        PART1_DIR / "Submission_Asset_Guide.txt",
        "Part 1 Submission Asset Guide",
        PART1_MAIN_FIGURES,
        PART1_SUPPLEMENTARY_FIGURES,
        PART1_MAIN_TABLE_SHEETS,
    )


def package_part2() -> None:
    figure_dir = PART2_DIR / "figures"
    ensure_multiformat_figures(figure_dir)
    rename_figure_files(figure_dir, PART2_RENAME_MAP)
    ensure_multiformat_figures(figure_dir)

    workbook_path = PART2_DIR / "tables" / "Part2_Anatomical_Evidence_Tables.xlsx"
    workbook_for_use = rewrite_workbook(workbook_path, PART2_SHEET_NAME_MAP)

    copy_figure_bundle(figure_dir, PART2_MAIN_FIGURES, PART2_DIR / "main_text_figures")
    copy_figure_bundle(figure_dir, PART2_SUPPLEMENTARY_FIGURES, PART2_DIR / "supplementary_figures")

    write_subset_workbook(workbook_for_use, PART2_DIR / "main_text_tables" / "Part2_Main_Text_Tables.xlsx", PART2_MAIN_TABLE_SHEETS)
    supp_tables_dir = PART2_DIR / "supplementary_tables"
    if supp_tables_dir.exists():
        shutil.rmtree(supp_tables_dir)

    write_asset_guide(
        PART2_DIR / "Submission_Asset_Guide.txt",
        "Part 2 Submission Asset Guide",
        PART2_MAIN_FIGURES,
        PART2_SUPPLEMENTARY_FIGURES,
        PART2_MAIN_TABLE_SHEETS,
    )


def main() -> None:
    package_part1()
    package_part2()
    build_combined_main_tables()
    print("Submission assets prepared.")


if __name__ == "__main__":
    main()
