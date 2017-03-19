import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import boxcox
from sklearn.metrics import log_loss

from utils.categorical_utils import process_with_lambda, cols, get_exp_lambda

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

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)

train_file = '../../data/redhoop/train.json'
test_file = '../../data/redhoop/test.json'


def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]


def load_train():
    return basic_preprocess(pd.read_json(train_file))


def load_test():
    return basic_preprocess(pd.read_json(test_file))


def process_outliers_lat_long(train_df, test_df):
    min_lat = 40
    max_lat = 41
    min_long = -74.1
    max_long = -73

    good_lat = (train_df[LATITUDE] < max_lat) & (train_df[LATITUDE] > min_lat)
    good_long = (train_df[LONGITUDE] < max_long) & (train_df[LONGITUDE] > min_long)

    train_df = train_df[good_lat & good_long]

    bed_lat = (test_df[LATITUDE] >= max_lat) | (test_df[LATITUDE] <= min_lat)
    bed_long = (test_df[LONGITUDE] >= max_long) | (test_df[LONGITUDE] <= min_long)
    test_df[LATITUDE][bed_lat] = train_df[LATITUDE].mean()
    test_df[LONGITUDE][bed_long] = train_df[LONGITUDE].mean()

    return train_df, test_df


def basic_preprocess(df):
    df['num_features'] = df[u'features'].apply(len)
    df['num_photos'] = df['photos'].apply(len)
    df['word_num_in_descr'] = df['description'].apply(lambda x: len(x.split(' ')))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    bc_price, tmp = boxcox(df['price'])
    df['bc_price'] = bc_price

    return df


def with_lambda_loss(df, k, f):
    col = MANAGER_ID
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"] + \
               cols(col, TARGET, TARGET_VALUES)

    train_df, test_df = split_df(df, 0.7)
    lamdba_f = get_exp_lambda(k, f)
    train_df, test_df = process_with_lambda(train_df, test_df, col, TARGET, TARGET_VALUES, lamdba_f)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=1000, objective='mlogloss')
    # estimator = RandomForestClassifier(n_estimators=1000)
    estimator.fit(train_arr, train_target)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()


    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)
    loss = log_loss(test_target, proba)
    return loss




def get_the_best_loss(trials):
    try:
        return trials.best_trial['result']['loss']
    except:
        return None

#('k=19.0_f=0.197468244166.json', 0.59239640555959361), ('k=15.0_f=0.14119444578.json', 0.5932402918244617), ('k=32.0_f=0.256206446874.json', 0.59326921056798809), ('k=26.0_f=0.173356531492.json', 0.59369840464968526)


def do_test(runs, fldr, params):
    df = load_train()
    for (k,f) in params:
        l=[]
        print 'Running for k={}, f={}'.format(k,f)
        for i in range(runs):
            loss = with_lambda_loss(df.copy(),k,f)
            print '#{}'.format(i)
            print loss
            print 'mean={}'.format(np.mean(l))
            l.append(loss)
            with open(os.path.join(fldr, 'k={},f={}'.format(k,f)), 'w+') as fl:
                json.dump(l,fl)




params = [(19.0, 0.197468244166), (15.0, 0.14119444578), (32.0, 0.256206446874), (26.0, 0.173356531492)]
do_test(200, '/home/dpetrovskyi/PycharmProjects/kaggle/trash/test_perspective_mngr', params)