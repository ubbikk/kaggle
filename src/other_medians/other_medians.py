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


def process_other_mngr_medians(train_df, test_df):
    features = ['num_features', 'num_photos', 'word_num_in_descr']
    df = pd.concat([train_df, test_df])
    new_cols = []
    for f in features:
        col = 'get_by_mngr_{}_mean'.format(f)
        df[col] = df.groupby(MANAGER_ID)[f].transform('mean')
        new_cols.append(col)

        col = 'get_by_mngr_{}_median'
        new_cols.append(col)
        df[col] = df.groupby(MANAGER_ID)[f].transform('median')

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols