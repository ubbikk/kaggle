import pandas as pd
TARGET='interest_level'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'

def explore_target_on_val(df, col, val):
    return pd.get_dummies(df[[TARGET, col]][df[col] == val], columns=[TARGET]).mean()

def explore_col_values_counts(df, col, N=None):
    res = df.groupby(col)[col].count().sort_values(ascending=False)
    if N is None:
        N=len(res)
    return res.head(N)

def explore_outliers_lat_long(df):
    min_lat=40
    max_lat=41
    min_long=-74.1
    max_long=-73

    bed_lat = (df[LATITUDE] >= max_lat) | (df[LATITUDE] <= min_lat)
    bed_long = (df[LONGITUDE] >= max_long) | (df[LONGITUDE] <= min_long)

    return df[bed_lat | bed_long]



