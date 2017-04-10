from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

TARGET = u'interest_level'
TARGET_VALUES = ['low', 'medium', 'high']
MANAGER_ID = 'manager_id'
BUILDING_ID = 'building_id'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
PRICE = 'price'
BATHROOMS = 'bathrooms'
BEDROOMS = 'bedrooms'
DESCRIPTION = 'description'
DISPLAY_ADDRESS = 'display_address'
STREET_ADDRESS = 'street_address'
LISTING_ID = 'listing_id'
PRICE_PER_BEDROOM = 'price_per_bedroom'
F_COL=u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
NEI = 'neighbourhood'
BORO = 'boro'
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'
BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'


def process_mngr_avg_median_price(train_df, test_df):
    quantiles = [0.1*x for x in range(1, 10)]
    df = pd.concat([train_df, test_df])
    bed_bath_median = 'bed_bath_median'
    df[bed_bath_median] = df.groupby([BED_NORMALIZED, BATH_NORMALIZED])[PRICE].transform('median')

    bed_bath_diff = 'bed_bath_diff'
    df[bed_bath_diff]=df[PRICE]-df[bed_bath_median]

    bed_bath_raio = 'bed_bath_ratio'
    df[bed_bath_raio]=df[bed_bath_diff]/df['bed_bath_median']

    cols_to_quantile = ['bed_bath_diff', 'bed_bath_ratio',
                        PRICE, LATITUDE, LONGITUDE, BED_NORMALIZED, BATH_NORMALIZED,
                        'num_features', 'num_photos', 'word_num_in_descr']
    new_cols=[bed_bath_median, bed_bath_diff, bed_bath_raio]

    group_by = df.groupby(MANAGER_ID)
    for col in cols_to_quantile:
        new_col = 'gr_by_mngr_mean_of_{}'.format(col)
        df[new_col] = group_by[col].transform('mean')
        new_cols.append(new_col)
        for q in quantiles:
            new_col = 'gr_by_mngr_{}_quantile_of_{}'.format(q, col)
            df[new_col] = group_by[col].transform('quantile', q)
            new_cols.append(new_col)


    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols