from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SignFlipReport:
    flipped_cols: List[str]
    untouched_cols: List[str]
    missing_cols: List[str]
    # per-column counts: (n_total, n_nonnull, n_neg_before, n_pos_before, n_zero_before)
    summary: Dict[str, Tuple[int, int, int, int, int]]


def signflip_negative_metrics(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    inplace: bool = False,
    only_if_mostly_negative: bool = False,
    mostly_negative_threshold: float = 0.5,  # e.g., flip only if >50% of non-null values are negative
    require_numeric: bool = True,
) -> Tuple[pd.DataFrame, SignFlipReport]:
    """
    Multiply selected columns by -1 (sign-flip) to convert 'negative metrics' into 'positive direction'
    metrics, optionally only when a column is mostly negative.

    Returns (df_out, report).
    """
    df_out = df if inplace else df.copy()

    cols = list(cols)
    missing = [c for c in cols if c not in df_out.columns]
    present = [c for c in cols if c in df_out.columns]

    flipped, untouched = [], []
    summary: Dict[str, Tuple[int, int, int, int, int]] = {}

    for c in present:
        s = df_out[c]

        if require_numeric and not pd.api.types.is_numeric_dtype(s):
            untouched.append(c)
            summary[c] = (len(s), s.notna().sum(), int((s < 0).sum()) if pd.api.types.is_numeric_dtype(s) else 0,
                          int((s > 0).sum()) if pd.api.types.is_numeric_dtype(s) else 0,
                          int((s == 0).sum()) if pd.api.types.is_numeric_dtype(s) else 0)
            continue

        n_total = len(s)
        n_nonnull = int(s.notna().sum())
        n_neg = int((s < 0).sum())
        n_pos = int((s > 0).sum())
        n_zero = int((s == 0).sum())

        summary[c] = (n_total, n_nonnull, n_neg, n_pos, n_zero)

        if n_nonnull == 0:
            untouched.append(c)
            continue

        if only_if_mostly_negative:
            frac_neg = n_neg / n_nonnull
            if frac_neg <= mostly_negative_threshold:
                untouched.append(c)
                continue

        df_out[c] = -s
        flipped.append(c)

    report = SignFlipReport(
        flipped_cols=flipped,
        untouched_cols=untouched,
        missing_cols=missing,
        summary=summary,
    )
    return df_out, report

def verify_signflip_invariants(
    before: pd.DataFrame,
    after: pd.DataFrame,
    flipped_cols: Iterable[str],
    *,
    atol: float = 1e-10
) -> pd.DataFrame:
    rows = []
    for c in flipped_cols:
        b = before[c]
        a = after[c]

        # Absolute values preserved (ignoring NaNs)
        abs_ok = np.allclose(np.abs(b.dropna().to_numpy()),
                             np.abs(a.dropna().to_numpy()),
                             atol=atol, rtol=0)

        # Mean flips sign, std preserved (population vs sample doesn’t matter for equality check)
        mean_ok = np.isclose(a.mean(skipna=True), -b.mean(skipna=True), atol=atol, rtol=0)
        std_ok  = np.isclose(a.std(skipna=True),  b.std(skipna=True),  atol=atol, rtol=0)

        # Correlation with itself should be exactly -1 if there is variance (otherwise NaN)
        corr = b.corr(a)

        rows.append({
            "col": c,
            "abs_preserved": abs_ok,
            "mean_flipped": mean_ok,
            "std_preserved": std_ok,
            "corr_before_after": corr
        })
    return pd.DataFrame(rows)
