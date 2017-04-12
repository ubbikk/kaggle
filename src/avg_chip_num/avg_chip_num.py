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

def process_avg_chip_num(train_df, test_df):
    df = pd.concat([train_df, test_df])
    new_cols=[]

    tmp = 'tmp'
    df[tmp] = df['median_ratio'].apply(lambda s: 1 if s<=0 else 0)

    col ='mngr_cheap_rent'
    groupby_manager_id_tmp_ = df.groupby(MANAGER_ID)[tmp]
    df[col] = groupby_manager_id_tmp_.transform('mean')
    new_cols.append(col)

    col ='mngr_cheap_num'
    df[col] = groupby_manager_id_tmp_.transform('sum')
    new_cols.append(col)

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols