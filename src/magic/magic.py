import pandas as pd


magic_file = '../../data/redhoop/listing_image_time.csv'
MAGIC_COL = 'magic'
LISTING_ID = 'listing_id'

def process_magic(train_df, test_df):
    m = pd.read_csv(magic_file)
    df = pd.concat([train_df, test_df])
    df = pd.merge(df, m, left_on=LISTING_ID, right_on='Listing_Id')
    new_cols = [MAGIC_COL]
    df[MAGIC_COL]=df['time_stamp']

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols
