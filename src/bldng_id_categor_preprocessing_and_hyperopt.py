import json
import os
from time import time

import seaborn as sns
import pandas as pd
from collections import OrderedDict

from hyperopt import STATUS_FAIL
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import tpe
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
from hyperopt import hp, pyll, fmin
from math import log

src_folder = '/home/ubik/PycharmProjects/kaggle/src'
os.chdir(src_folder)
import sys

sys.path.append(src_folder)

from categorical_utils import process_with_lambda, cols, get_exp_lambda, visualize_exp_lambda
from v2w import avg_vector_df, load_model, avg_vector_df_and_pca

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


def with_lambda_loss(df, k, f):
    col = BUILDING_ID
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


def loss_for_batch(df, s,trials, runs, flder):
    t = time()

    f=s['f']
    k=s['k']
    if k<=1 or f<=0.1:
        return {'loss': 1000,  'status': STATUS_FAIL}

    # print 'Running for k={}, f={}'.format(k,f)
    l = []
    fp = os.path.join(flder, 'k={}_f={}.json'.format(k, f))
    for x in range(runs):
        loss = with_lambda_loss(df.copy(), k, f)
        l.append(loss)
        with open(fp, 'w+') as fl:
            json.dump(l, fl)

    t = time() - t
    # print loss

    avg_loss = np.mean(l)
    var = np.var(l)

    print '\n\n'
    print 'summary for k={}, f={}'.format(k,f)
    print 'current_loss={}, best={}'.format(avg_loss, get_the_best_loss(trials))
    print
    print 'time: {}'.format(t)
    print 'std={}'.format(np.std(l))
    print '\n\n'

    return {'loss': avg_loss, 'loss_variance': var, 'status': STATUS_OK}


def get_the_best_loss(trials):
    try:
        return trials.best_trial['result']['loss']
    except:
        return None


def do_test(runs, fldr):
    df = load_train()
    space = {
        'k': hp.qnormal('k', 25, 10, 1),
        'f': hp.loguniform('f', log(0.1), log(5))
    }
    trials = Trials()
    best = fmin(lambda s: loss_for_batch(df,s,trials, runs,fldr), space=space, algo=tpe.suggest, trials=trials, max_evals=10000)

    print
    print 'curr={}'.format(best)
    print 'best={}'.format(get_the_best_loss(trials))




do_test(25, '/home/ubik/PycharmProjects/kaggle/trash/building_id_exp_family_hyperopt_25')