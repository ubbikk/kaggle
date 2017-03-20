from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
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

def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)

def get_dummy_cols(col_name, col_values):
    return ['{}_{}'.format(col_name, val) for val in col_values]

def process_test_train_clustering_df(train_df, test_df, N):
    new_col = 'clustered_{}'.format(MANAGER_ID)
    tmp = pd.get_dummies(train_df[[MANAGER_ID, TARGET]], columns=[TARGET])
    tmp=tmp.groupby(MANAGER_ID).mean()
    cols = get_dummy_cols(TARGET, TARGET_VALUES)
    X = tmp[cols]

    est = MiniBatchKMeans(n_clusters=N)
    tmp[new_col] = est.fit_predict(X)

    train_df=pd.merge(train_df, tmp, left_on=MANAGER_ID, right_index=True)
    test_df=pd.merge(test_df, tmp, left_on=MANAGER_ID, right_index=True, how='left')
    test_df[test_df[new_col].isnull()]=N

    return train_df, test_df, [new_col]

