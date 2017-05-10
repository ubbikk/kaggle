import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

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
CREATED = "created"
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
DAY_OF_YEAR='day_of_year'
LABEL = 'label'
SECONDS='seconds'
BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'
BED_BATH_MEDIAN= 'bed_bath_median'
BED_BATH_DIFF = 'bed_bath_diff'
BED_BATH_RATIO = 'bed_bath_ratio'

def process_mngr_weighted_price_ratio(train_df, test_df):
    return process_weighted_price_ratio(train_df, test_df, MANAGER_ID, 5)



def process_weighted_price_ratio(train_df, test_df, col, folds):
    skf = StratifiedKFold(folds)
    target_vals = ['high', 'medium', 'low']
    new_cols = ['weighted_price_ratio', 'weighted_price_diff']
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        calc_weighted_price_ratio(big, small, col, train_df)

    calc_weighted_price_ratio(train_df.copy(), test_df.copy(), col, update_df=test_df)

    return train_df, test_df, new_cols


def calc_weighted_price_ratio(big, small, col, update_df):
    target_vals = ['high', 'medium', 'low']
    new_cols = ['weighted_price_ratio', 'weighted_price_diff']

    big['target_cp'] = big[TARGET].copy()
    big= pd.get_dummies(big, columns=['target_cp'])
    big['weighted_price_ratio'] = 3*big['target_cp_high']*big[BED_BATH_RATIO]+big['target_cp_medium']*big[BED_BATH_RATIO]
    big['weighted_price_diff'] = 3*big['target_cp_high']*big[BED_BATH_DIFF]+big['target_cp_medium']*big[BED_BATH_DIFF]

    grouped = big.groupby(col).mean()
    small = pd.merge(small[[col]], grouped[new_cols], left_on=col, right_index=True)
    for new_col in new_cols:
        update_df.loc[small.index, new_col] = small[new_col]