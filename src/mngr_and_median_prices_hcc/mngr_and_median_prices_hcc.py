import pandas as pd

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


def process_mngr_and_median_ratio_hcc(train_df, test_df):
    df = pd.concat([train_df, test_df])

    mngr_plus_minus = 'mngr_plus_minus'
    df[mngr_plus_minus] = df['median_ratio'].apply(lambda s: 'minus' if s<=0 else 'plus')
    df[mngr_plus_minus] = df[MANAGER_ID]+'_'+df[mngr_plus_minus]

    train_df[mngr_plus_minus]=df.loc[train_df.index, mngr_plus_minus]
    test_df[mngr_plus_minus]=df.loc[test_df.index, mngr_plus_minus]

    col = mngr_plus_minus

    new_cols = []
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols