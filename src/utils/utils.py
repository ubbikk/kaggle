import pandas as pd
TARGET='interest_level'

def explore_target_on_val(df, col, val):
    return pd.get_dummies(df[[TARGET, col]][df[col] == val], columns=[TARGET]).mean()

def explore_col_values_counts(df, col, N=None):
    res = df.groupby(col)[col].count().sort_values(ascending=False)
    if N is None:
        N=len(res)
    return res.head(N)
