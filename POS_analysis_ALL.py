"""
Independent Script: POS Auxiliary Analysis (POS→OS Correlation Following Non-linear Correction in ELN)
================================================================
- Fully standalone operation; contains no code from the "main model (ELN-centric)".
- Objective: Maximise **elimination of console alerts/noise** without compromising results.
- Stability: Forced numerisation of Age/Year; merging of rare levels by category; Cox L2 regularisation + dropout; optional stratification (strata).
- Details: RMST calculation independent of 'KM_estimate' column name; uses `np.trapezoid` to avoid deprecation warnings; PH test output redirected to file.
*** Translated with www.DeepL.com/Translator (free version) ***
"""
from __future__ import annotations

import os
import io
import contextlib
import argparse
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.errors import SettingWithCopyWarning
from lifelines import CoxPHFitter, KaplanMeierFitter
try:
    from lifelines.utils import ConvergenceWarning, StatisticalWarning  # lifelines≥0.28
except Exception:  # 兼容旧版 lifelines 无 StatisticalWarning 的情况
    from lifelines.utils import ConvergenceWarning
    StatisticalWarning = UserWarning  # 回落为 UserWarning，依旧可被 warnings.filterwarnings 屏蔽
from lifelines.exceptions import ConvergenceError
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import chi2

# 可选：时间依赖 AUC（若未安装将自动跳过）
try:
    from sksurv.util import Surv
    from sksurv.metrics import cumulative_dynamic_auc
    HAVE_SKSURV = True
except Exception:
    HAVE_SKSURV = False

# —— 全局静音（不影响结果，仅抑制控制台噪声） ——
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=StatisticalWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# -------------------------------
# 实用函数
# -------------------------------
SEER_MAP = {
    'SurvM': 'Survival months',
    'Event': 'Vital status recode (study cutoff used)',
    'POS': 'Regional nodes positive (1988+)',
    'ELN': 'Regional nodes examined (1988+)',
    'Primary_part': 'Primary Site - labeled',
    'Stage': 'Summary stage 2000 (1998-2017)',
    'SurgeryExtent': 'RX Summ--Surg Prim Site (1998+)',
    'Age': 'Age',
    'Sex': 'Sex',
    'Grade': 'Grade Recode (thru 2017)',
    'Year': 'Year of diagnosis',
}


def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


def ensure_outdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lo, hi)


def _coerce_numeric_str(x: pd.Series) -> pd.Series:
    """将形如 '27 years' / '27岁' / '27' 的字符串转为数字；不可解析为 NaN。"""
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors='coerce')
    return pd.to_numeric(x.astype(str).str.extract(r'(-?\d+(?:\.\d+)?)')[0], errors='coerce')


def _collapse_rare_levels(s: pd.Series, min_count: int = 20, missing_label: str = 'Missing') -> pd.Series:
    """将出现次数过少的水平合并为 '__RARE__'，并将缺失填为 missing_label。"""
    s = s.astype(str).fillna(missing_label)
    vc = s.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    return s.where(~s.isin(rare), other='__RARE__')


def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # 基础列映射
    if 'SurvM' not in d.columns and SEER_MAP['SurvM'] in d.columns:
        d['SurvM'] = pd.to_numeric(d[SEER_MAP['SurvM']], errors='coerce')
    if 'Event' not in d.columns and SEER_MAP['Event'] in d.columns:
        ev = d[SEER_MAP['Event']].astype(str).str.lower()
        d['Event'] = (ev.str.contains('dead') | ev.str.contains('死亡')).astype(int)
    if 'POS' not in d.columns and SEER_MAP['POS'] in d.columns:
        d['POS'] = pd.to_numeric(d[SEER_MAP['POS']], errors='coerce')
    if 'ELN' not in d.columns and SEER_MAP['ELN'] in d.columns:
        d['ELN'] = pd.to_numeric(d[SEER_MAP['ELN']], errors='coerce')
    # LNR（若缺失则派生）
    if 'LNR' not in d.columns and {'POS','ELN'}.issubset(d.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            d['LNR'] = d['POS']/d['ELN']
            d['LNR'] = d['LNR'].replace([np.inf, -np.inf], np.nan)
    # Age / Year 尽量转为数值
    if 'Age' in d.columns:
        d['Age'] = _coerce_numeric_str(d['Age'])
    if 'Year' in d.columns:
        d['Year'] = _coerce_numeric_str(d['Year'])
    return d


def prepare_dataset(df: pd.DataFrame,
                    min_follow_months: float = 0.0,
                    winsorize_pos: Tuple[float, float] = (0.01, 0.99),
                    winsorize_eln: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    d = auto_map_columns(df)

    need = ['SurvM', 'Event', 'POS', 'ELN']
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise ValueError(f"Missing key columns：{miss}")

    # 强制数值化
    d['SurvM'] = pd.to_numeric(d['SurvM'], errors='coerce')
    d['Event'] = pd.to_numeric(d['Event'], errors='coerce').fillna(0).astype(int)
    d['POS'] = pd.to_numeric(d['POS'], errors='coerce')
    d['ELN'] = pd.to_numeric(d['ELN'], errors='coerce')
    if 'Age' in d.columns:
        d['Age'] = _coerce_numeric_str(d['Age'])
    if 'Year' in d.columns:
        d['Year'] = _coerce_numeric_str(d['Year'])

    # 基本过滤
    mask = (
        d['SurvM'].notna() & d['Event'].notna() & (d['SurvM'] > min_follow_months)
        & d['POS'].notna() & d['ELN'].notna() & (d['ELN'] > 0)
    )
    d = d.loc[mask].copy()

    # 温莎化极端值
    if winsorize_pos is not None:
        d['POS'] = winsorize_series(d['POS'], *winsorize_pos)
    if winsorize_eln is not None:
        d['ELN'] = winsorize_series(d['ELN'], *winsorize_eln)

    # 派生变量：POS_minus1
    d['POS_minus1'] = (d['POS'] - 1).clip(lower=0)

    # 协变量候选与清洗
    candidate_covars = [
        'Age', 'Sex', 'Primary_part', 'Stage', 'SurgeryExtent', 'Grade', 'Year',
        'chemo_bin', 'radiation_bin'
    ]
    present_covars = [c for c in candidate_covars if c in d.columns]

    # 数值缺失以中位数填补；分类型缺失合并为 Missing，并合并稀有水平
    for v in present_covars:
        if v in ['Age', 'Year']:
            d[v] = pd.to_numeric(d[v], errors='coerce')
            d[v] = d[v].fillna(d[v].median())
        else:
            d[v] = _collapse_rare_levels(d[v], min_count=20, missing_label='Missing')

    d.attrs['present_covars'] = present_covars

    if 'LNR' not in d.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            d['LNR'] = d['POS']/d['ELN']
            d['LNR'] = d['LNR'].replace([np.inf, -np.inf], np.nan)
    return d


# -------------------------------
# 模型 M2：ELN RCS 校正下的 POS 关联（主辅助模型）
# -------------------------------

def build_formula_pos_rcs(present_covars: List[str], df_pos: int = 3, df_eln: int = 3,
                          add_interaction: bool = False,
                          strata_cols: Optional[List[str]] = None) -> str:
    if strata_cols is None:
        strata_cols = []
    terms = ["I(POS > 0)", f"cr(POS_minus1, {df_pos})", f"cr(ELN, {df_eln})"]
    for v in present_covars:
        if v in strata_cols:
            continue  # 分层变量不再进入固定效应
        if v in ['Age', 'Year']:
            terms.append(v)
        else:
            terms.append(f"C({v})")
    if add_interaction:
        terms.append("I(POS > 0):ELN")
    return ' + '.join(terms)


def fit_pos_model(df: pd.DataFrame, outdir: str,
                  df_pos: int = 3, df_eln: int = 3,
                  add_interaction: bool = False,
                  robust: bool = True,
                  penalizer: float = 0.1,
                  strata_cols: Optional[List[str]] = None) -> CoxPHFitter:
    set_chinese_font(); ensure_outdir(outdir)
    if strata_cols is None:
        strata_cols = []
    present_covars = df.attrs.get('present_covars', [])
    formula = build_formula_pos_rcs(present_covars, df_pos, df_eln, add_interaction, strata_cols)

    # 带 L2 正则以缓解多重共线/近零方差
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(df, duration_col='SurvM', event_col='Event', formula=formula, robust=robust,
                strata=strata_cols if len(strata_cols) > 0 else None)
    except ConvergenceError:
        cph = CoxPHFitter(penalizer=max(1.0, penalizer))
        cph.fit(df, duration_col='SurvM', event_col='Event', formula=formula, robust=robust,
                strata=strata_cols if len(strata_cols) > 0 else None)

    coef_ = cph.summary.copy()
    coef_.to_excel(os.path.join(outdir, 'T_POS_main_model_summary.xlsx'))

    plot_pos_effect_curves(cph, df, outdir)
    check_ph_and_save(cph, df, outdir)

    with open(os.path.join(outdir, 'MODEL_FORMULA.txt'), 'w', encoding='utf-8') as f:
        f.write('Formula: ' + formula + '\n')
        if strata_cols:
            f.write('Strata : ' + ', '.join(strata_cols) + '\n')
    return cph


def plot_pos_effect_curves(cph: CoxPHFitter, df: pd.DataFrame, outdir: str,
                           pos_grid: Optional[List[int]] = None):
    set_chinese_font()
    if pos_grid is None:
        # 结合分位数与经验点，避免右尾过稀疏
        if (df['POS'] > 0).sum() > 0:
            q = df.loc[df['POS'] > 0, 'POS'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round().astype(int)
            pos_grid = sorted(set([1, 2, 3, 5, 10] + q.tolist()))
            pos_grid = [p for p in pos_grid if p >= 1 and p <= int(df['POS'].max())]
        else:
            pos_grid = [1, 2, 3, 4, 5]

    rng = np.random.default_rng(2025)
    idx = rng.choice(df.index, size=min(800, len(df)), replace=False)
    baseX = df.loc[idx].copy()

    eln_med = float(df['ELN'].median())

    X0 = baseX.copy(); X0['POS'] = 0; X0['POS_minus1'] = 0; X0['ELN'] = eln_med
    X1 = baseX.copy(); X1['POS'] = 1; X1['POS_minus1'] = 0; X1['ELN'] = eln_med
    hr_step = float(cph.predict_partial_hazard(X1).mean() / cph.predict_partial_hazard(X0).mean())

    hrs, vals = [], []
    for p in pos_grid:
        Xp = baseX.copy(); Xp['POS'] = p; Xp['POS_minus1'] = max(p - 1, 0); Xp['ELN'] = eln_med
        ph = cph.predict_partial_hazard(Xp).mean()
        hr = float(ph / cph.predict_partial_hazard(X0).mean())
        hrs.append(hr); vals.append(p)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax[0].bar([0, 1], [1.0, hr_step], width=0.6)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['POS=0', 'POS≥1'])
    ax[0].set_ylabel('Relative risk (HR)')
    ax[0].set_title('Positive/Negative Transition HR (ELN Fixed Median)')
    ax[0].axhline(1.0, ls='--', lw=1, c='k')

    ax[1].plot(vals, hrs, marker='o')
    ax[1].set_xlabel('Number of positive lymph nodes (POS)')
    ax[1].set_ylabel('Relative risk (HR, reference POS=0)')
    ax[1].set_title('Spline HR curve with POS>0 (ELN fixed median)')
    ax[1].axhline(1.0, ls='--', lw=1, c='k')

    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'F_POS1_HRcurve.png'), dpi=220); plt.close()


def check_ph_and_save(cph: CoxPHFitter, df: pd.DataFrame, outdir: str):
    report_path = os.path.join(outdir, 'PH_check_POS.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('Summary of Proportional Hazard (PH) Assumption Testing\n')
        f.write('=====================================\n')
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cph.check_assumptions(df, show_plots=True, p_value_threshold=0.05)
        plt.savefig(os.path.join(outdir, 'F_POS_PH_Schoenfeld.png'), dpi=200)
        plt.close()
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write('Schoenfeld Residual plot saved：F_POS_PH_Schoenfeld.png\n')
            f.write('\n【lifelines check_assumptions Output has been redirected as follows】\n\n')
            f.write(buf.getvalue())
    except Exception as e:
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f'check_assumptions Warning/Exception：{e}\n')


# -------------------------------
# 可选：ELN×POS 交互可视化
# -------------------------------

def plot_pos_by_eln_strata(cph: CoxPHFitter, df: pd.DataFrame, outdir: str,
                            eln_low: Optional[float] = None, eln_high: Optional[float] = None,
                            pos_vals: Optional[List[int]] = None):
    set_chinese_font(); ensure_outdir(outdir)
    if pos_vals is None:
        pos_vals = [1, 2, 3, 4, 5, 8, 10]
        pos_vals = [p for p in pos_vals if p <= int(df['POS'].max())]
    if eln_low is None:
        eln_low = float(df['ELN'].quantile(0.25))
    if eln_high is None:
        eln_high = float(df['ELN'].quantile(0.75))

    rng = np.random.default_rng(2025)
    idx = rng.choice(df.index, size=min(800, len(df)), replace=False)
    baseX = df.loc[idx].copy()

    X0_low = baseX.copy(); X0_low['POS'] = 0; X0_low['POS_minus1'] = 0; X0_low['ELN'] = eln_low
    X0_high = baseX.copy(); X0_high['POS'] = 0; X0_high['POS_minus1'] = 0; X0_high['ELN'] = eln_high
    ref_low = cph.predict_partial_hazard(X0_low).mean()
    ref_high = cph.predict_partial_hazard(X0_high).mean()

    vals, hr_low, hr_high = [], [], []
    for p in pos_vals:
        Xl = baseX.copy(); Xl['POS'] = p; Xl['POS_minus1'] = max(p - 1, 0); Xl['ELN'] = eln_low
        Xh = baseX.copy(); Xh['POS'] = p; Xh['POS_minus1'] = max(p - 1, 0); Xh['ELN'] = eln_high
        hr_l = float(cph.predict_partial_hazard(Xl).mean() / ref_low)
        hr_h = float(cph.predict_partial_hazard(Xh).mean() / ref_high)
        vals.append(p); hr_low.append(hr_l); hr_high.append(hr_h)

    plt.figure(figsize=(6.8, 4.8))
    plt.plot(vals, hr_low, marker='o', label=f'ELN lower decile（~{eln_low:.0f}）')
    plt.plot(vals, hr_high, marker='s', label=f'ELN High percentile（~{eln_high:.0f}）')
    plt.axhline(1.0, ls='--', c='k', lw=1)
    plt.xlabel('Number of positive lymph nodes（POS）'); plt.ylabel('Relative risk（HR）')
    plt.title('POS–HR Curve under the ELN Hierarchy')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'F_POS2_POS_by_ELN_interaction.png'), dpi=220); plt.close()


# -------------------------------
# 增量信息（可选）：L1 vs L2
# -------------------------------

def fit_incremental(df: pd.DataFrame, outdir: str,
                    df_eln: int = 3, df_lnr: int = 3, df_pos: int = 3,
                    robust: bool = True,
                    time_points: List[int] = [12, 36, 60]) -> pd.DataFrame:
    set_chinese_font(); ensure_outdir(outdir)
    present_covars = df.attrs.get('present_covars', [])

    def covar_terms():
        ts = []
        for v in present_covars:
            if v in ['Age', 'Year']:
                ts.append(v)
            else:
                ts.append(f"C({v})")
        return ' + '.join(ts)

    base_cov = covar_terms()
    f_L1 = ' + '.join([f"cr(ELN, {df_eln})", f"cr(LNR, {df_lnr})", base_cov]).strip(' +')
    f_L2 = ' + '.join([f_L1, "I(POS > 0)", f"cr(POS_minus1, {df_pos})"]).strip(' +')

    c1 = CoxPHFitter(penalizer=0.1); c1.fit(df, 'SurvM', 'Event', formula=f_L1, robust=robust)
    c2 = CoxPHFitter(penalizer=0.1); c2.fit(df, 'SurvM', 'Event', formula=f_L2, robust=robust)

    ll1, ll2 = c1.log_likelihood_, c2.log_likelihood_
    df_diff = c2.params_.shape[0] - c1.params_.shape[0]
    LR = 2 * (ll2 - ll1); p_LRT = 1 - chi2.cdf(LR, df_diff)

    out = {
        'model_L1': f_L1,
        'model_L2': f_L2,
        'loglik_L1': ll1,
        'loglik_L2': ll2,
        'LR': LR,
        'df_diff': df_diff,
        'p_LRT': p_LRT,
        'C_index_L1': c1.concordance_index_,
        'C_index_L2': c2.concordance_index_,
        'delta_C': c2.concordance_index_ - c1.concordance_index_,
    }

    # （可选）时间依赖 AUC
    if HAVE_SKSURV:
        y = Surv.from_arrays(event=df['Event'].astype(bool).to_numpy(), time=df['SurvM'].to_numpy())
        s1 = -c1.predict_partial_hazard(df).to_numpy().reshape(-1)
        s2 = -c2.predict_partial_hazard(df).to_numpy().reshape(-1)
        for tau in time_points:
            times = np.array([tau], dtype=float)
            auc1, _ = cumulative_dynamic_auc(y, y, s1, times)
            auc2, _ = cumulative_dynamic_auc(y, y, s2, times)
            out[f'AUC_{tau}m_L1'] = float(auc1[0])
            out[f'AUC_{tau}m_L2'] = float(auc2[0])
            out[f'delta_AUC_{tau}m'] = float(auc2[0] - auc1[0])

    res = pd.DataFrame([out])
    res.to_excel(os.path.join(outdir, 'T_POS_increment.xlsx'), index=False)
    return res


# -------------------------------
# POS 分档：重叠权重(OW) + 加权 KM + ΔRMST
# -------------------------------

def pos_tier(x: float) -> str:
    if x <= 0: return 'POS=0'
    if 1 <= x <= 2: return 'POS=1–2'
    if 3 <= x <= 4: return 'POS=3–4'
    return 'POS≥5'


def rmst_from_km(kmf: KaplanMeierFitter, tau: float) -> float:
    sf = kmf.survival_function_
    # 列名可能是自定义 label，而非 'KM_estimate'，取首列更稳健
    col = sf.columns[0] if hasattr(sf, 'columns') and len(sf.columns) > 0 else 'KM_estimate'
    t = sf.index.to_numpy()
    S = sf[col].to_numpy()
    # 线性外推到 tau
    if tau > t[-1]:
        t = np.append(t, tau); S = np.append(S, S[-1])
    else:
        mask = t <= tau
        t = t[mask]; S = S[mask]
        if t[-1] < tau:
            t = np.append(t, tau); S = np.append(S, S[-1])
    # 使用 numpy.trapezoid 以避免弃用告警
    return float(np.trapezoid(S, t))


def binary_ow_weights(X: pd.DataFrame, z: np.ndarray) -> np.ndarray:
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=200))
    clf.fit(X, z)
    e = clf.predict_proba(X)[:, 1]
    w = np.where(z == 1, 1.0 - e, e)
    return w


def weighted_km_and_rmst_by_pos(df: pd.DataFrame, outdir: str,
                                time_points: List[int] = [12, 36, 60],
                                balance_covars: Optional[List[str]] = None,
                                eln_rcs_df: int = 3) -> pd.DataFrame:
    set_chinese_font(); ensure_outdir(outdir)
    d = df.copy(); d['POS_tier'] = d['POS'].apply(pos_tier)

    if balance_covars is None:
        balance_covars = df.attrs.get('present_covars', [])

    def design_matrix(data: pd.DataFrame) -> pd.DataFrame:
        terms = [f"cr(ELN, {eln_rcs_df})"]
        for v in balance_covars:
            if v in ['Age', 'Year']:
                terms.append(v)
            else:
                terms.append(f"C({v})")
        f = ' + '.join(terms) if terms else '1'
        X = dmatrix(f, data, return_type='dataframe')
        return X

    X_all = design_matrix(d)

    basel = d[d['POS_tier'] == 'POS=0']
    tiers = ['POS=1–2', 'POS=3–4', 'POS≥5']

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    rows = []
    for tier in tiers:
        sub = pd.concat([basel, d[d['POS_tier'] == tier]], axis=0)
        z = (sub['POS_tier'] == tier).astype(int).to_numpy()
        X = X_all.loc[sub.index]

        w = binary_ow_weights(X, z)

        km_c = KaplanMeierFitter(label=f'{tier} vs 0：POS=0').fit(
            sub[z == 0]['SurvM'], sub[z == 0]['Event'], weights=w[z == 0]
        )
        km_t = KaplanMeierFitter(label=f'{tier} vs 0：{tier}').fit(
            sub[z == 1]['SurvM'], sub[z == 1]['Event'], weights=w[z == 1]
        )

        km_t.plot(ax=ax)

        for tau in time_points:
            rmst_c = rmst_from_km(km_c, tau)
            rmst_t = rmst_from_km(km_t, tau)
            rows.append({
                'contrast': f'{tier} vs POS=0', 'tau_m': tau,
                'RMST_t': rmst_t, 'RMST_c': rmst_c,
                'Delta_RMST': rmst_t - rmst_c
            })

    ax.set_title('POS classification: Weighted KM (overlap weighting; RCS correction including ELN)）')
    ax.set_xlabel('Follow-up (monthly)'); ax.set_ylabel('Survival rate')
    ax.legend(loc='best'); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'F_POS3_KM_POS_tiers.png'), dpi=220); plt.close()

    tab = pd.DataFrame(rows)
    tab.to_excel(os.path.join(outdir, 'T_POS3_DeltaRMST.xlsx'), index=False)
    return tab


# -------------------------------
# 主入口（独立运行）
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description='POS Auxiliary Analysis (Standalone Script · Silent Edition)')
    parser.add_argument('--excel', type=str, default='Pancreatic_Body_Tail_ALL.xlsx', help='Enter the Excel file path')
    parser.add_argument('--sheet', type=str, default='export', help='Worksheet name')
    parser.add_argument('--outdir', type=str, default='outputs_ALL_POS', help='Output directory')
    parser.add_argument('--df_pos', type=int, default=3, help='POS spline degrees of freedom')
    parser.add_argument('--df_eln', type=int, default=3, help='ELN spline degrees of freedom (for calibration)')
    parser.add_argument('--interaction', type=int, default=0, help='Should we implement ELN×POS stratified visualisation? (0/1)')
    parser.add_argument('--do_incremental', type=int, default=0, help='Whether to run incremental information L1 vs L2 (0/1)')
    parser.add_argument('--do_weighted_km', type=int, default=1, help='Should weighted KM be run with ΔRMST(0/1)?')
    parser.add_argument('--strata', type=str, default='', help='A comma-separated list of hierarchical variables，example "SurgeryExtent,Stage"')
    parser.add_argument('--penalizer', type=float, default=0.1, help='Cox L2 Regularisation Strength (Increase if convergence is unstable)')
    args = parser.parse_args()

    set_chinese_font(); ensure_outdir(args.outdir)

    # 读取数据
    if not os.path.exists(args.excel):
        raise FileNotFoundError(f'Not Found Excel：{args.excel}')
    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    d = prepare_dataset(df)

    strata_cols = [s.strip() for s in args.strata.split(',') if s.strip()] if args.strata else []

    # 主辅助模型（仅 POS 相关；含 ELN 非线性校正）
    cph = fit_pos_model(d, outdir=args.outdir, df_pos=args.df_pos, df_eln=args.df_eln,
                        add_interaction=bool(args.interaction), robust=True,
                        penalizer=args.penalizer, strata_cols=strata_cols)

    # 可选：ELN×POS 分层可视化
    if args.interaction:
        plot_pos_by_eln_strata(cph, d, args.outdir)

    # 可选：增量信息（与 LNR 锚定）
    if args.do_incremental:
        fit_incremental(d, args.outdir, df_eln=args.df_eln, df_lnr=3, df_pos=args.df_pos, robust=True)

    # 可选：加权 KM + ΔRMST（POS 分档）
    if args.do_weighted_km:
        weighted_km_and_rmst_by_pos(d, args.outdir, time_points=[12, 36, 60])

    print(f"Completed. Output directory：{args.outdir}")


if __name__ == '__main__':
    main()
