MANAGER_ID = 'manager_id'
TARGET = u'interest_level'
BATHROOMS = 'bathrooms'
BEDROOMS = 'bedrooms'

def process_mngr_group_col_hcc(train_df, test_df, col):
    col_to_group = 'mngr_id_group_{}'.format(col)
    for df in (train_df, test_df):
        df[col_to_group] = df[MANAGER_ID] + '_' + df[col]

    new_cols = []
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col_to_group, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols

def process_group_bed_bath(train_df, test_df):
    new_cols=[]
    train_df, test_df, cls = process_mngr_group_col_hcc(train_df, test_df, BEDROOMS)
    new_cols+=cls
    train_df, test_df, cls = process_mngr_group_col_hcc(train_df, test_df, BATHROOMS)
    new_cols+=cls

    return train_df, test_df, new_cols