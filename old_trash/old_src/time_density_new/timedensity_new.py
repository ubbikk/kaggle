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
F_COL = u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE = 'created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
CREATED='created'
LABEL='lbl'
TOTAL_MINUTES='total_minutes'
TOTAL_10_MINUTES='total_10_minutes'
TOTAL_60_MINUTES='total_60_minutes'

def process_time_density_new(train_df, test_df):
    df = pd.concat([train_df, test_df])
    df[TOTAL_MINUTES] = df[CREATED_MINUTE]+24*df[CREATED_HOUR]
    df[TOTAL_10_MINUTES] = df[TOTAL_MINUTES].apply(lambda s: s/10)
    df[TOTAL_60_MINUTES] = df[TOTAL_MINUTES].apply(lambda s: s/60)

    new_cols=[]

    for col in [TOTAL_MINUTES, TOTAL_10_MINUTES, TOTAL_60_MINUTES]:
        n_col = 'in_{}_num'.format(col)
        df[n_col] = df.groupby(col)[MANAGER_ID].transform('count')
        new_cols.append(n_col)

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols


def process_avg_mngr_time(train_df, test_df):
    df = pd.concat([train_df, test_df])
    features = ['in_{}_num'.format(col) for col in [TOTAL_MINUTES, TOTAL_10_MINUTES, TOTAL_60_MINUTES]]
    new_cols = []
    for f in features:
        col = 'get_by_mngr_{}_mean'.format(f)
        df[col] = df.groupby(MANAGER_ID)[f].transform('mean')
        new_cols.append(col)

        col = 'get_by_mngr_{}_median'.format(f)
        new_cols.append(col)
        df[col] = df.groupby(MANAGER_ID)[f].transform('median')

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols

