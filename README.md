# Examined Lymph Node Thresholds and Surgical Approach in Body/Tail Pancreatic Ductal Adenocarcinoma

Analysis code and reproducibility materials for the manuscript investigating:
- Optimal examined lymph node (ELN) thresholds for survival in pancreatic body/tail cancer
- Anatomical evidence synthesis for surgical approach families (standard DP, RAMPS, retroperitoneal/posterior)
- Theoretical modeling of surgical accessibility (Mattox Maneuver framework)

## Repository Structure

```
├── part1_seer_pipeline_v3.py          # Part 1: SEER cohort analysis (main pipeline)
├── part2_anatomical_evidence_pipeline.py # Part 2: Anatomical evidence synthesis
├── part1_seer_pipeline.py             # Part 1: SEER analysis (legacy v1)
├── prepare_submission_assets.py       # Submission asset preparation
├── submission_prep_safe.py            # Safe submission preparation helper
├── mattox_models.py                   # Theoretical modeling (Mattox framework)
├── model_results.json                 # Theoretical model parameters
├── validation_results.json            # Model validation results
├── rmst_bootstrap_results.csv         # RMST bootstrap analysis results
├── rmst_bootstrap_pnorm.csv           # RMST bootstrap p-norm results
├── ELN_analysis_ALL_v2.py             # ELN analysis (extended)
├── POS_analysis_ALL.py                # Positive node count analysis
└── outputs/
    ├── part1_seer_analysis/           # SEER analysis outputs (v3, final)
    │   ├── figures/                   # Main and supplementary figures
    │   ├── tables/                    # Analysis tables
    │   └── V3_KM_Risk_Table_Metadata.json
    └── part2_anatomical_evidence/     # Anatomical evidence outputs
        ├── figures/                   # Approach-specific figures
        └── tables/                    # Evidence summary tables
```

## Data Availability

The SEER (Surveillance, Epidemiology, and End Results) data used in this study are **not included** in this repository due to the SEER Data-Use Agreement. Researchers can obtain SEER Research Plus data after completing the data-use agreement at:
- https://seer.cancer.gov/data/

The analysis pipeline expects the following SEER data files (not included):
- `Pancreatic_Body_Tail_ALL.xlsx` — SEER cohort export

## Requirements

```
Python >= 3.8
pandas
numpy
scipy
statsmodels
lifelines
scikit-learn
matplotlib
seaborn
openpyxl
python-docx
```

Install dependencies:
```bash
pip install pandas numpy scipy statsmodels lifelines scikit-learn matplotlib seaborn openpyxl python-docx
```

## Reproducibility

### Part 1: SEER Cohort Analysis

```bash
# Place SEER data file in the working directory
python part1_seer_pipeline_v3.py
```

This produces:
- Descriptive statistics (Table 1)
- ELN threshold search results (Table 2)
- Cox proportional hazards models
- RMST bootstrap analysis
- Restricted cubic splines
- Sensitivity analyses (POS, LNR, LODDS)

### Part 2: Anatomical Evidence Synthesis

```bash
python part2_anatomical_evidence_pipeline.py
```

This produces:
- Approach-family ELN distributions (Table 3)
- Multi-study forest plots
- Threshold evidence maps
- Theoretical accessibility modeling

### Theoretical Modeling

```bash
python mattox_models.py
```

This produces:
- Counterfactual accessibility analysis
- Survival benefit decomposition
- Monte Carlo staging validation

## Key Results

- **ELN threshold ≥12**: Significant survival benefit (RMST gain 2.59 months, 95% CI 1.13–3.92)
- **ELN threshold ≥21**: Additional benefit in IPTW analysis (RMST gain 2.91 months, 95% CI 1.33–4.22)
- **Surgical approach**: RAMPS attainment rate for ELN≥21 was 40.0% (95% CI 35.0–45.0) via Gaussian approximation with continuity correction

## Citation

If you use this code, please cite the corresponding manuscript.

## License

This analysis code is provided for reproducibility purposes. See the manuscript for data use restrictions.
