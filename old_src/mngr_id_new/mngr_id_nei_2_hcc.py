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
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'
NEI = 'neighbourhood'
BORO = 'boro'

# def process_mngr_categ_preprocessing(train_df, test_df):
#     col = MANAGER_ID
#     new_cols = []
#     for df in [train_df, test_df]:
#         df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
#         df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
#     for binary_col in ['target_high', 'target_medium']:
#         train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
#         new_cols.append(new_col)
#
#     return train_df, test_df, new_cols


def process_mngr_and_nei2(train_df, test_df):
    col = 'mngr_nei2'
    for df in (train_df, test_df):
        df[col] = df[MANAGER_ID] + '_' + df[NEI_2]

    new_cols = []
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols