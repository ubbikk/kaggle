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
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
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
PRICE_PER_BEDROOM = 'price_per_bedroom'
F_COL = u'features'

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)

train_file = '../../data/redhoop/train.json'
test_file = '../../data/redhoop/test.json'


def out(l, loss, num, t):
    print '\n\n'
    print '#{}'.format(num)
    print 'loss {}'.format(loss)
    print
    print 'avg_loss {}'.format(np.mean(l))
    print 'std {}'.format(np.std(l))
    print 'time {}'.format(t)


def write_results(l, fp):
    with open(fp, 'w+') as f:
        json.dump(l, f)


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


# (0.61509489625789615, [0.61124170916042475, 0.61371758902339113, 0.61794752159334343, 0.61555861194203254, 0.61700904957028924])
def simple_loss(df):
    month_dunmies = ['created_month_4', 'created_month_5', 'created_month_6']
    days_dumies = ['created_day_1',
                   'created_day_2', 'created_day_3', 'created_day_4', 'created_day_5',
                   'created_day_6', 'created_day_7', 'created_day_8', 'created_day_9',
                   'created_day_10', 'created_day_11', 'created_day_12', 'created_day_13',
                   'created_day_14', 'created_day_15', 'created_day_16', 'created_day_17',
                   'created_day_18', 'created_day_19', 'created_day_20', 'created_day_21',
                   'created_day_22', 'created_day_23', 'created_day_24', 'created_day_25',
                   'created_day_26', 'created_day_27', 'created_day_28', 'created_day_29',
                   'created_day_30', 'created_day_31']
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr'] + month_dunmies + days_dumies
    # print df.columns.values

    train_df, test_df = split_df(df, 0.7)

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
    return log_loss(test_target, proba)


def loss_with_per_tree_stats(df):
    month_dunmies = ['created_month_4', 'created_month_5', 'created_month_6']
    days_dumies = ['created_day_1',
                   'created_day_2', 'created_day_3', 'created_day_4', 'created_day_5',
                   'created_day_6', 'created_day_7', 'created_day_8', 'created_day_9',
                   'created_day_10', 'created_day_11', 'created_day_12', 'created_day_13',
                   'created_day_14', 'created_day_15', 'created_day_16', 'created_day_17',
                   'created_day_18', 'created_day_19', 'created_day_20', 'created_day_21',
                   'created_day_22', 'created_day_23', 'created_day_24', 'created_day_25',
                   'created_day_26', 'created_day_27', 'created_day_28', 'created_day_29',
                   'created_day_30', 'created_day_31']
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_n'
                                              'um_in_descr'] + month_dunmies + days_dumies

    train_df, test_df = split_df(df, 0.7)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=1500, objective='mlogloss')
    # estimator = RandomForestClassifier(n_estimators=1000)
    eval_set = [(train_arr, train_target), (test_arr, test_target)]
    estimator.fit(train_arr, train_target, eval_set=eval_set, eval_metric='mlogloss', verbose=False)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()


    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)

    return log_loss(test_target, proba), xgboost_per_tree_results(estimator)


def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    results_on_train = estimator.evals_result()['validation_0']['mlogloss']
    return {
        'train': results_on_train,
        'test': results_on_test
    }


def do_test(num, fp):
    l = []
    train_df = load_train()
    train_df = pd.get_dummies(train_df, columns=['created_month', "created_day"])
    for x in range(num):
        t = time()
        df = train_df.copy()

        loss = simple_loss(df)
        t = time() - t
        l.append(loss)

        out(l, loss, x, t)
        write_results(l, fp)


def do_test_with_xgboost_stats_per_tree(num, fp):
    l = []
    results = []
    train_df = load_train()
    train_df = pd.get_dummies(train_df, columns=['created_month', "created_day"])
    for x in range(num):
        t = time()
        df = train_df.copy()

        loss, res = loss_with_per_tree_stats(df)
        t = time() - t
        l.append(loss)
        results.append(res)

        out(l, loss, x, t)
        write_results(results, fp)


do_test_with_xgboost_stats_per_tree(1000, 'month_and_day.json')
