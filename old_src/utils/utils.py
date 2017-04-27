import pandas as pd
import numpy as np
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

def show_target_freq_deviations(n):
    interest_level_high  =    0.077788
    interest_level_low   =   0.694683
    interest_level_medium =   0.227529

    arr = np.random.random(n)
    h = len(filter(lambda s: s<interest_level_high, arr))
    m = len(filter(lambda s: s>=interest_level_high and s<interest_level_high+interest_level_medium, arr))
    l = len(filter(lambda s: s>=interest_level_high+interest_level_medium, arr))

    print 'high     {}'.format(h*1.0/n)
    print 'medium   {}'.format(m*1.0/n)
    print 'low      {}'.format(l*1.0/n)




