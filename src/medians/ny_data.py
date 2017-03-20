import json
import pandas as pd
from collections import OrderedDict

from scipy.spatial import KDTree

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
DISTANCE_TO_NEAREST_NEIGBOUR = 'dist_n'
ZIPCODE='zip_code'
BORO = 'boro'
NEI = 'nei'

zip_map_file='medians/zip_by_nei_and_boro_corrected.json'
ny_data_file = '/home/dpetrovskyi/PycharmProjects/kaggle/src/neighbothood/ny.csv'

def load_zip_map():
    return json.load(open(zip_map_file))

def load_ny_data():
    return pd.read_csv(ny_data_file)

def add_fields_from_nearest_neighbour(df):
    ny_data = load_ny_data()
    tree = KDTree(ny_data[[LATITUDE, LONGITUDE]].values)
    res = tree.query(df[[LATITUDE, LONGITUDE]].values)
    df[DISTANCE_TO_NEAREST_NEIGBOUR] = res[0]
    df['neigh_index'] = res[1]

    df[ZIPCODE] = df['neigh_index'].apply(lambda s: ny_data.loc[s, ZIPCODE])
    df.loc[df[ZIPCODE].isnull(), ZIPCODE]=-999
    df[ZIPCODE]=df[ZIPCODE].apply(int)
    df[ZIPCODE]=df[ZIPCODE].apply(str)

    zip_map = load_zip_map()
    df[BORO]=df[ZIPCODE].apply(lambda s: zip_map[s][BORO])
    df[NEI]=df[ZIPCODE].apply(lambda s: zip_map[s][NEI])

    return df

#add_fields_from_nearest_neighbour(train_df)
#train_df, test_df = split_df(train_df, 0.7)
#train_df, test_df, new_cols = process_price_mean(train_df, test_df)


def process_price_mean(train_df, test_df):
    avg_price = train_df[PRICE].mean()
    agg = OrderedDict([(PRICE, {'avg_price': 'mean'}), (LATITUDE, {'count': 'count'})])
    df = train_df.groupby(NEI).agg(agg)
    new_col = 'avg_price_diff'
    df.columns = [new_col, 'count']
    df.loc[df['count'] < 100, new_col]=avg_price

    train_df = pd.merge(train_df, df, left_on=NEI, right_index=True)
    train_df[new_col] = train_df[new_col]-train_df[PRICE]

    test_df = pd.merge(test_df, df, left_on=NEI, right_index=True)
    test_df.loc[test_df[new_col].isnull(), new_col] = avg_price
    test_df[new_col] = test_df[new_col]-test_df[PRICE]

    return train_df, test_df, [new_col]
