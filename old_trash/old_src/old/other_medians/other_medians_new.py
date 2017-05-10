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

def get_main_value(s):
    n = int(0.66*len(s))
    vals = {k:0 for k in set(s)}
    for x in s:
        vals[x]+=1

    for k,v in vals.iteritems():
        if v>=n:
            return k



def process_other_mngr_medians_new(train_df, test_df):
    df = pd.concat([train_df, test_df])
    total_minutes_col='total_minutes'
    df[total_minutes_col] = df[CREATED_MINUTE]+24*df[CREATED_HOUR]
    features = [PRICE, LATITUDE, LONGITUDE, total_minutes_col]
    new_cols = []
    for f in features:
        col = 'get_by_mngr_{}_mean'.format(f)
        df[col] = df.groupby(MANAGER_ID)[f].transform('mean')
        new_cols.append(col)

        col = 'get_by_mngr_{}_median'.format(f)
        new_cols.append(col)
        df[col] = df.groupby(MANAGER_ID)[f].transform('median')

    main_hour='main_hour'
    bl = df.groupby(MANAGER_ID)[CREATED_HOUR].apply(get_main_value).to_frame(main_hour)
    df = pd.merge(df, bl, left_on=MANAGER_ID, right_index=True)
    new_cols.append(main_hour)

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]




    return train_df, test_df, new_cols