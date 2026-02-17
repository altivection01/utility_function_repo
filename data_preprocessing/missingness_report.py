# import pandas as pd

def missing_report(df, sort_by='missing', ascending=False, pct_decimals=2):
    n_rows = len(df)

    report = pd.DataFrame({
        'Missing': df.isna().sum(),
        'Sparsity %': (df.isna().sum() / n_rows * 100).round(pct_decimals),
        'dtype': df.dtypes,
        'Example': [df[col].dropna().iloc[0] if df[col].notna().any() else '-' for col in df.columns]
    }).rename_axis('Column')
    
    sort_col = 'Missing' if sort_by == 'missing' else 'Sparsity %' if sort_by == 'pct' else None
    
    if sort_col:
        report = report.sort_values(sort_col, ascending=ascending)
    return report
