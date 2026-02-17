import pandas as pd 

def clean_categorical_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Clean multiple categorical columns in a DataFrame by stripping
    whitespace and converting values to uppercase.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        List of column names to clean.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned categorical columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df_cleaned: pd.DataFrame = df.copy()

    for col in columns:
        if col not in df_cleaned.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        df_cleaned[col] = clean_categorical_series(df_cleaned[col])

    return df_cleaned
