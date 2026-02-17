# requires pandas
def normalize_metrics(df, metric_cols, denominator_col):
    df = df.copy()

    # safe denominator: 0 → NaN to avoid inf
    denom = df[denominator_col].replace(0, np.nan)

    for col in metric_cols:
        rate_col = f"{col}_rate"

        # base division
        df[rate_col] = df[col] / denom

        # explicit rule: if numerator == 0, rate == 0
        df.loc[df[col] == 0, rate_col] = 0

    return df
