import json
import os
from time import time

import seaborn as sns
import pandas as pd
from collections import OrderedDict

from matplotlib import pyplot
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import cross_val_score, KFold
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from scipy.stats import boxcox
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
# PRICE_PER_BEDROOM = 'price_per_bedroom'

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
    min_lat=40
    max_lat=41
    min_long=-74.1
    max_long=-73

    good_lat = (train_df[LATITUDE] < max_lat) & (train_df[LATITUDE] > min_lat)
    good_long = (train_df[LONGITUDE] < max_long) & (train_df[LONGITUDE] > min_long)

    train_df = train_df[good_lat & good_long]

    bed_lat = (test_df[LATITUDE] >=max_lat) | (test_df[LATITUDE] <=min_lat)
    bed_long = (test_df[LONGITUDE] >= max_long) | (test_df[LONGITUDE] <= min_long)
    test_df[LATITUDE][bed_lat] = train_df[LATITUDE].mean()
    test_df[LONGITUDE][bed_long]=train_df[LONGITUDE].mean()

    return train_df, test_df

# pricePerBed=ifelse(!is.finite(price/bedrooms),-1, price/bedrooms)
# ,pricePerBath=ifelse(!is.finite(price/bathrooms),-1, price/bathrooms)
# ,pricePerRoom=ifelse(!is.finite(price/(bedrooms+bathrooms)),-1, price/(bedrooms+bathrooms))
# ,bedPerBath=ifelse(!is.finite(bedrooms/bathrooms), -1, price/bathrooms)
# ,bedBathDiff=bedrooms-bathrooms
# ,bedBathSum=bedrooms+bathrooms
# ,bedsPerc=ifelse(!is.finite(bedrooms/(bedrooms+bathrooms)), -1
PRICE_PER_BED='pricePerBed'
PRICE_PER_BATH='pricePerBath'
PRICE_PER_ROOM='pricePerRoom'
BED_PER_BATH='bedPerBath'
BED_BATH_DIFF='bedBathDiff'
BED_BATH_SUM='bedBathSum'
BEDS_PER_C='bedsPerc'

ADDITIONAL_BED_BATH_FEATURES=[
    PRICE_PER_BED,
PRICE_PER_BATH,
PRICE_PER_ROOM,
BED_PER_BATH,
BED_BATH_DIFF,
BED_BATH_SUM,
BEDS_PER_C,
]

def guarded_ratio(x, n1, n2):
    if(x[n2]==0):
        return -1
    return float(x[n1])/x[n2]

def two_cols_ratio(df, col1, col2, new_col):
    df[new_col]=df.apply(lambda x: guarded_ratio(x, col1, col2), axis=1)

def add_additional_bed_bathrooms_feature(train_df, test_df):
    for df in [train_df, test_df]:
        # df[BEDROOMS]= df[BEDROOMS].apply(float)
        two_cols_ratio(df, PRICE, BEDROOMS, PRICE_PER_BED)
        two_cols_ratio(df, PRICE, BATHROOMS, PRICE_PER_BATH)
        df[BED_BATH_SUM] = df[BATHROOMS]+df[BEDROOMS]
        two_cols_ratio(df, PRICE, BED_BATH_SUM, PRICE_PER_ROOM)
        two_cols_ratio(df, BEDROOMS, BATHROOMS, BED_PER_BATH)
        df[BED_BATH_DIFF]= df[BEDROOMS]-df[BATHROOMS]
        two_cols_ratio(df, BEDROOMS, BED_BATH_SUM, BEDS_PER_C)


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


# (0.61509489625789615, [0.61124170916042475, 0.61371758902339113, 0.61794752159334343, 0.61555861194203254, 0.61700904957028924])
def simple_loss(df):
    additional_features = [PRICE_PER_BATH, PRICE_PER_BED, PRICE_PER_ROOM]
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]+additional_features


    train_df, test_df = split_df(df, 0.7)
    add_additional_bed_bathrooms_feature(train_df, test_df)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=1000, objective='mlogloss')
    estimator.fit(train_arr, train_target)
    proba = estimator.predict_proba(test_arr)
    return log_loss(test_target, proba)


def do_test(num, fp):
    l = []
    train_df = load_train()
    for x in range(num):
        t=time()
        df=train_df.copy()
        loss = simple_loss(df)
        print '\n\n'
        print 'loss {}'.format(loss)
        print
        print 'avg loss {}'.format(np.mean(l))
        print 'std {}'.format(np.std(l))
        print 'time: {}'.format(time()-t)
        l.append(loss)
        with open(fp, 'w+') as f:
            json.dump(l, f)

    print '\n\n\n\n'
    print 'avg = {}'.format(np.mean(l))

# train_df, test_df = load_train(), load_test()
# add_additional_bed_bathrooms_feature(train_df, test_df)

do_test(1000, '/home/dpetrovskyi/PycharmProjects/kaggle/src/bath_bedrooms_features/price_per_bath_bed_room.json')
