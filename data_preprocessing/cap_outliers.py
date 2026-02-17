import numpy as np
import pandas as pd


def cap_outliers(
    series: pd.Series,
    method: str = "percentile",
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
    iqr_multiplier: float = 1.5,
) -> pd.Series:
    """
    Cap outliers in a pandas Series using either percentile-based
    winsorization or IQR-based Tukey fencing.

    Parameters
    ----------
    series : pd.Series
        Numeric pandas Series to cap.
    method : str, default="percentile"
        Method to use: "percentile" or "iqr".
    lower_pct : float, default=0.01
        Lower percentile (used only if method="percentile").
    upper_pct : float, default=0.99
        Upper percentile (used only if method="percentile").
    iqr_multiplier : float, default=1.5
        Multiplier for IQR fences (used only if method="iqr").

    Returns
    -------
    pd.Series
        Series with capped outliers.

    Raises
    ------
    ValueError
        If method is not "percentile" or "iqr".
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    clean_series: pd.Series = series.astype(float)

    if method == "percentile":
        lower_bound: float = float(clean_series.quantile(lower_pct))
        upper_bound: float = float(clean_series.quantile(upper_pct))

    elif method == "iqr":
        q1: float = float(clean_series.quantile(0.25))
        q3: float = float(clean_series.quantile(0.75))
        iqr: float = q3 - q1

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

    else:
        raise ValueError("Method must be either 'percentile' or 'iqr'.")

    capped_series: pd.Series = clean_series.clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    print(
        f"Capping bounds → lower={lower_bound:.4f}, "
        f"upper={upper_bound:.4f}"
    )

    return capped_series
