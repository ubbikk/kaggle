import json
import os
from collections import OrderedDict
from math import log
from time import time

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_FAIL
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp, fmin
from hyperopt import tpe
from hyperopt.mongoexp import MongoTrials
from scipy.stats import boxcox
from sklearn.metrics import log_loss
from functools import partial
import math
import mngr_id_categorical_optimizer
import sys


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


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)

train_file = '../data/redhoop/train.json'
test_file = '../data/redhoop/test.json'


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




def get_the_best_loss(trials):
    try:
        return trials.best_trial['result']['loss']
    except:
        return None


def do_test(runs, flder):
    df = load_train()
    space = {
        'k': hp.qnormal('k', 25, 10, 1),
        'f': hp.loguniform('f', log(0.1), log(5))
    }
    trials = MongoTrials('mongo://10.20.0.144:27017/redhop_mngr_id_exp_family/jobs', exp_key='exp1')

    log_file = os.path.join(flder, 'log.txt')
    objective = partial(mngr_id_categorical_optimizer.loss_for_batch, df=df, runs=runs, flder=flder, log_file=log_file)
    best = fmin(objective, space=space, algo=tpe.suggest, trials=trials,
                max_evals=10000)

    print
    print 'curr={}'.format(best)
    print 'best={}'.format(get_the_best_loss(trials))


do_test(2, sys.argv[1])
