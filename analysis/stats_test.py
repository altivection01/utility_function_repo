from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    levene,
    mannwhitneyu,
    shapiro,
    ttest_ind,
)

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
except ImportError:  # pragma: no cover
    sm = None
    smf = None
    anova_lm = None


def _as_clean_series(x: Union[pd.Series, Sequence[float]]) -> pd.Series:
    """
    Coerce input to a float Series and drop NaNs.

    Parameters
    ----------
    x : Union[pd.Series, Sequence[float]]
        Input values.

    Returns
    -------
    pd.Series
        Float series with NaNs dropped.
    """
    return pd.Series(x, dtype="float64").dropna()


def _shapiro_pvalue(series: pd.Series, alpha: float) -> Tuple[float, bool]:
    """
    Compute Shapiro-Wilk p-value on up to 5000 samples.

    Parameters
    ----------
    series : pd.Series
        Input series (NaNs should already be dropped).
    alpha : float
        Significance threshold for declaring non-normality.

    Returns
    -------
    Tuple[float, bool]
        (p_value, is_normal)
    """
    n: int = int(min(len(series), 5000))
    if n < 3:
        return float("nan"), False

    stat, p_value = shapiro(series.sample(n=n, random_state=42))
    _ = stat
    return float(p_value), bool(p_value > alpha)


def _levene_pvalue(groups: List[pd.Series], alpha: float) -> Tuple[float, Optional[bool]]:
    """
    Compute median-centered Levene test across 2+ groups.

    Parameters
    ----------
    groups : List[pd.Series]
        List of groups (NaNs should already be dropped).
    alpha : float
        Significance threshold for declaring unequal variances.

    Returns
    -------
    Tuple[float, Optional[bool]]
        (p_value, equal_variance_assumed). If not enough data, returns (nan, None).
    """
    if len(groups) < 2:
        return float("nan"), None

    if any(len(g) < 2 for g in groups):
        return float("nan"), None

    stat, p_value = levene(*groups, center="median")
    _ = stat
    return float(p_value), bool(p_value > alpha)


def run_stat_test(
    group1: Union[pd.Series, Sequence[Any]],
    group2: Union[pd.Series, Sequence[Any]],
    test_name: str = "ttest",
    alpha: float = 0.05,
    levene_alpha: float = 0.05,
    *,
    data: Optional[pd.DataFrame] = None,
    dv: Optional[str] = None,
    factors: Optional[List[str]] = None,
    include_interactions: bool = True,
    anova_typ: int = 2,
    robust: bool = False,
) -> Dict[str, Any]:
    """
    Unified hypothesis test utility supporting:
      - Two-sample tests: "ttest", "anova" (2 groups), "chi2"
      - Multifactor ANOVA: "anova_multi" using statsmodels OLS + anova_lm

    For two-sample mean tests:
      - Shapiro-Wilk normality check (sample up to 5000)
      - Levene homogeneity-of-variance check (median-centered)
      - Fallbacks:
          * ttest -> Welch if variances unequal
          * ttest -> Mann-Whitney U if non-normal
          * anova (2 groups) -> Welch if variances unequal

    For multifactor ANOVA:
      - Fits OLS with categorical factors (treated as C(...))
      - Reports ANOVA table
      - Normality check on residuals (Shapiro, sampled)
      - Levene across cell groups (all factor combinations)
      - Optionally computes heteroskedasticity-robust term tests (robust=True)

    Parameters
    ----------
    group1, group2 : Union[pd.Series, Sequence[Any]]
        For two-sample tests: numeric vectors (ttest/anova) or categorical vectors (chi2).
        For "anova_multi", group1/group2 are ignored; provide `data`, `dv`, `factors`.
    test_name : str, default="ttest"
        One of: "ttest", "anova", "chi2", "anova_multi".
    alpha : float, default=0.05
        Significance threshold.
    levene_alpha : float, default=0.05
        Levene test threshold for equal variances assumption.
    data : Optional[pd.DataFrame], keyword-only
        Required for "anova_multi".
    dv : Optional[str], keyword-only
        Dependent variable column name for "anova_multi".
    factors : Optional[List[str]], keyword-only
        Factor column names for "anova_multi".
    include_interactions : bool, default=True, keyword-only
        Include all interaction terms between factors for "anova_multi".
    anova_typ : int, default=2, keyword-only
        Type of sums of squares for ANOVA table (commonly 2 or 3).
    robust : bool, default=False, keyword-only
        If True for "anova_multi", also compute robust (HC3) term tests.

    Returns
    -------
    Dict[str, Any]
        Standardized result dictionary with test outputs and diagnostics.

    Raises
    ------
    ValueError
        For invalid test_name or missing required inputs.
    """
    if test_name not in {"ttest", "anova", "chi2", "anova_multi"}:
        raise ValueError("test_name must be one of: 'ttest', 'anova', 'chi2', 'anova_multi'")

    if test_name == "anova_multi":
        if smf is None or anova_lm is None:
            raise ValueError(
                "statsmodels is required for 'anova_multi'. Install it: pip install statsmodels"
            )

        if data is None or dv is None or factors is None or len(factors) < 1:
            raise ValueError("For 'anova_multi', you must provide data, dv, and factors (len>=1).")

        needed_cols: List[str] = [dv] + factors
        missing_cols: List[str] = [c for c in needed_cols if c not in data.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"Missing required columns in data: {missing_cols}")

        df_model: pd.DataFrame = data[needed_cols].dropna().copy()

        # Build formula: dv ~ C(f1) + C(f2) + C(f1):C(f2) + ...
        factor_terms: List[str] = [f"C({f})" for f in factors]
        if include_interactions and len(factor_terms) > 1:
            rhs: str = " * ".join(factor_terms)
        else:
            rhs = " + ".join(factor_terms)

        formula: str = f"{dv} ~ {rhs}"

        model = smf.ols(formula=formula, data=df_model).fit()

        # Residual normality check (Shapiro on sampled residuals)
        resid_series: pd.Series = pd.Series(model.resid, dtype="float64").dropna()
        resid_shapiro_p, resid_normal = _shapiro_pvalue(resid_series, alpha=alpha)

        # Levene across all cell groups (factor combinations)
        group_cols: List[str] = factors
        grouped = df_model.groupby(group_cols, dropna=True)[dv]
        cell_groups: List[pd.Series] = [_as_clean_series(v) for _, v in grouped]

        levene_p, equal_var = _levene_pvalue(cell_groups, alpha=levene_alpha)

        anova_table: pd.DataFrame = anova_lm(model, typ=anova_typ)

        robust_terms: Optional[pd.DataFrame] = None
        if robust:
            robust_res = model.get_robustcov_results(cov_type="HC3")
            try:
                wt = robust_res.wald_test_terms(skip_single=False)
                robust_terms = wt.table
            except Exception:
                robust_terms = None

        # Determine "significance" as: any term p < alpha (excluding Residual)
        pvals = anova_table.get("PR(>F)")
        any_sig: bool = False
        if pvals is not None:
            pvals_clean = pvals.drop(labels=["Residual"], errors="ignore").dropna()
            any_sig = bool((pvals_clean < alpha).any())

        return {
            "test_used": "Multifactor ANOVA (OLS)",
            "formula": formula,
            "alpha": float(alpha),
            "stat": float("nan"),
            "p_value": float("nan"),
            "significant": any_sig,
            "normality_p_group1": float("nan"),
            "normality_p_group2": float("nan"),
            "residual_normality_p": float(resid_shapiro_p),
            "residuals_normal": bool(resid_normal),
            "levene_p": float(levene_p),
            "equal_variance_assumed": equal_var,
            "anova_table": anova_table,
            "robust_term_tests": robust_terms,
            "n_rows_used": int(len(df_model)),
        }

    # ---- Two-sample modes below ----
    if test_name == "chi2":
        ct: pd.DataFrame = pd.crosstab(pd.Series(group1), pd.Series(group2))
        stat, p_value, dof, expected = chi2_contingency(ct)

        return {
            "test_used": "Chi-square",
            "alpha": float(alpha),
            "stat": float(stat),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "normality_p_group1": float("nan"),
            "normality_p_group2": float("nan"),
            "levene_p": float("nan"),
            "equal_variance_assumed": None,
            "dof": int(dof),
            "expected": expected,
            "contingency_table": ct,
        }

    g1: pd.Series = _as_clean_series(group1)
    g2: pd.Series = _as_clean_series(group2)

    p_norm1, normal1 = _shapiro_pvalue(g1, alpha=alpha)
    p_norm2, normal2 = _shapiro_pvalue(g2, alpha=alpha)
    normal: bool = bool(normal1 and normal2)

    levene_p, equal_var = _levene_pvalue([g1, g2], alpha=levene_alpha)

    if test_name == "ttest":
        if normal:
            if equal_var is False:
                stat, p_value = ttest_ind(g1, g2, equal_var=False)
                test_used = "Welch's t-test (unequal variances)"
            else:
                stat, p_value = ttest_ind(g1, g2, equal_var=True)
                test_used = "Independent t-test (equal variances)"
        else:
            stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")
            test_used = "Mann-Whitney U (non-parametric)"

    else:  # "anova" with 2 groups
        if equal_var is False:
            stat, p_value = ttest_ind(g1, g2, equal_var=False)
            test_used = "Welch's t-test (used instead of ANOVA due to unequal variances)"
        else:
            stat, p_value = f_oneway(g1, g2)
            test_used = "One-way ANOVA (2 groups)"

    return {
        "test_used": test_used,
        "alpha": float(alpha),
        "stat": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "normality_p_group1": float(p_norm1),
        "normality_p_group2": float(p_norm2),
        "levene_p": float(levene_p),
        "equal_variance_assumed": equal_var,
        "n_group1": int(len(g1)),
        "n_group2": int(len(g2)),
    }
"""
EXAMPLE USAGE:
result = run_stat_test(
    group1=[],
    group2=[],
    test_name="anova_multi",
    alpha=0.05,
    levene_alpha=0.05,
    data=df,
    dv="late_aircraft_rate",
    factors=["month", "carrier"],
    include_interactions=True,
    anova_typ=2,
    robust=True,
)

print(f"Used: {result['test_used']}")
print(f"Residual normality p={result['residual_normality_p']:.4f}")
print(f"Levene p={result['levene_p']:.4f}")

anova_table = result["anova_table"]
print(anova_table)
"""
