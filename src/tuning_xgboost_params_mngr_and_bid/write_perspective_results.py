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
import math

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

def process_with_lambda(train_df, test_df, col, target_col, target_vals, lambda_f):
    temp_target = '{}_'.format(target_col)
    train_df[temp_target]= train_df[target_col]
    train_df= pd.get_dummies(train_df, columns=[target_col])
    dummies_cols = [dummy_col(target_col, v) for v in target_vals]
    priors = train_df[dummies_cols].mean()
    priors_arr = [priors[dummy_col(target_col, v)] for v in target_vals]
    agg = OrderedDict(
        [(dummy_col(target_col, v), OrderedDict([('{}_mean'.format(v),'mean')])) for v in target_vals] + [(col, {'cnt':'count'})]
    )
    df = train_df[[col]+dummies_cols].groupby(col).agg(agg)
    df.columns = ['posterior_{}'.format(v) for v in target_vals] + ['cnt']
    new_cols=[]
    for v in target_vals:
        def norm_posterior(x):
            cnt= float(x['cnt'])
            posterior = x['posterior_{}'.format(v)]
            prior = priors[dummy_col(target_col, v)]
            l = lambda_f(cnt)
            return (l * posterior) + ((1 - l) * prior)

        new_col = '{}_coverted_exp_for_{}={}'.format(col, target_col, v)
        df[new_col] =df.apply(norm_posterior, axis=1)
        new_cols.append(new_col)

    df = df[new_cols]

    train_df = pd.merge(train_df, df, left_on=col, right_index=True)

    test_df = pd.merge(test_df, df, left_on=col, right_index=True, how='left')
    test_df.loc[test_df[new_cols[0]].isnull(), new_cols] = priors_arr

    for c in dummies_cols:
        del train_df[c]

    train_df[target_col]= train_df[temp_target]
    del train_df[temp_target]

    return train_df, test_df, new_cols

def cols(col, target_col, target_vals):
    return ['{}_coverted_exp_for_{}={}'.format(col, target_col, v) for v in target_vals]

def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)


def mngr_and_bid_loss(df, params):

    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]

    train_df, test_df = split_df(df, 0.7)

    col = MANAGER_ID
    k=15.0
    f=0.14119444578
    lamdba_f = get_exp_lambda(k, f)
    train_df, test_df, new_cols = process_with_lambda(train_df, test_df, col, TARGET, TARGET_VALUES, lamdba_f)
    features+=new_cols

    col = BUILDING_ID
    k=51.0
    f=0.156103119211
    lamdba_f = get_exp_lambda(k, f)
    train_df, test_df, new_cols = process_with_lambda(train_df, test_df, col, TARGET, TARGET_VALUES, lamdba_f)
    features+=new_cols

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(
        n_estimators=1000,
        objective='mlogloss',
        subsample=params['subsample'],
        learning_rate=params['learning_rate'],
        colsample_bytree=params['colsample_bytree'],
        max_depth=params['max_depth']

    )
    estimator.fit(train_arr, train_target)

    proba = estimator.predict_proba(test_arr)
    loss = log_loss(test_target, proba)
    return loss

def get_exp_lambda(k,f):
    def res(n):
        return 1/(1+math.exp(float(k-n)/f))
    return res


# (0.61509489625789615, [0.61124170916042475, 0.61371758902339113, 0.61794752159334343, 0.61555861194203254, 0.61700904957028924])
def simple_loss(df, params):
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]

    train_df, test_df = split_df(df, 0.7)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(
        n_estimators=1000,
        objective='mlogloss',
        subsample=params['subsample'],
        learning_rate=params['learning_rate'],
        colsample_bytree=params['colsample_bytree'],
        max_depth=params['max_depth']

    )
    # estimator = RandomForestClassifier(n_estimators=1000)
    estimator.fit(train_arr, train_target)

    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)
    return log_loss(test_target, proba)


perspective_params = [
    {u'subsample': 0.6470208038499322, u'learning_rate': 0.03216888570403348, u'colsample_bytree': 0.8596250584891415,
     u'max_depth': 3},
    {u'subsample': 0.6484620540849879, u'learning_rate': 0.022519160081992555, u'colsample_bytree': 0.9888371794099271,
     u'max_depth': 4},
    {u'subsample': 0.6518087023497741, u'learning_rate': 0.022691097269614467, u'colsample_bytree': 0.8037309043218854,
     u'max_depth': 4},
    {u'subsample': 0.6011747553087955, u'learning_rate': 0.02025695974844694, u'colsample_bytree': 0.7544808142775762,
     u'max_depth': 4},
    {u'subsample': 0.6473852360121519, u'learning_rate': 0.024570185106733675, u'colsample_bytree': 0.9610129824750707,
     u'max_depth': 4}]

def do_test(num, fldr):
    for p in perspective_params:
        fl = 'subsample={},learning_rate={},colsample_bytree={},max_depth={}.json'.format(p['subsample'],
                                                                                                 p['learning_rate'],
                                                                                                 p['colsample_bytree'],
                                                                                                 p['max_depth'])
        fp=os.path.join(fldr,fl)
        test_params(num, fp,p)


def test_params(num, fp, params):
    neww = []
    train_df = load_train()
    for x in range(num):
        t = time()
        df = train_df.copy()
        loss = mngr_and_bid_loss(df, params)
        print loss
        print 'time: {}'.format(time() - t)
        print
        neww.append(loss)
        with open(fp, 'w+') as f:
            json.dump(neww, f)

    print '\n\n\n\n'
    print 'avg = {}'.format(np.mean(neww))


fldr= '../../trash/perspective_xgboost_params_results'

do_test(300, fldr)