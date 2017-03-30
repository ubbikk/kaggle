import os
import sys
from functools import partial
from math import log

import numpy as np
import pandas as pd
from hyperopt import Trials
from hyperopt import hp, fmin
from hyperopt import tpe
from hyperopt.mongoexp import MongoTrials
from scipy.stats import boxcox



import mngr_id_hcc_optimizer

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
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']


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
    # df["created_year"] = df["created"].dt.year
    df[CREATED_MONTH] = df["created"].dt.month
    df[CREATED_DAY] = df["created"].dt.day
    df[CREATED_HOUR] = df["created"].dt.hour
    df[CREATED_MINUTE] = df["created"].dt.minute
    df[DAY_OF_WEEK] = df['created'].dt.dayofweek
    bc_price, tmp = boxcox(df['price'])
    df['bc_price'] = bc_price

    return df




def get_the_best_loss(trials):
    try:
        return trials.best_trial['result']['loss']
    except:
        return None


def do_test(runs):
    df = load_train()
    space = {
        'mngr_k': hp.qnormal('mngr_k', 15, 11, 1),
        'mngr_f': hp.loguniform('mngr_f', log(0.1), log(5)),
        'mngr_n': hp.choice('mngr_n', [2,3,4,5,6,7,10]),

        'bid_k': hp.qnormal('bid_k', 15, 11, 1),
        'bid_f': hp.loguniform('bid_f', log(0.1), log(5)),
        'bid_n': hp.choice('bid_n', [2,3,4,5,6,7,10])
    }
    trials = MongoTrials('mongo://10.20.0.144:27017/mngr_id_hcc_08_08/jobs', exp_key='exp1')

    objective = partial(.loss_for_batch, df=df, runs=runs)
    best = fmin(objective, space=space, algo=tpe.suggest, trials=trials,
                max_evals=10000)

    print
    print 'curr={}'.format(best)
    print 'best={}'.format(get_the_best_loss(trials))


do_test(25)
