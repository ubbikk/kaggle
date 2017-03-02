import json
import os
import seaborn as sns
import pandas as pd
from collections import OrderedDict

from hyperopt import STATUS_OK, STATUS_FAIL
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
from hyperopt import hp, pyll, fmin


src_folder = '/home/dpetrovskyi/PycharmProjects/kaggle/src'
os.chdir(src_folder)
import sys
sys.path.append(src_folder)

from v2w import avg_vector_df, load_model, avg_vector_df_and_pca

TARGET = u'interest_level'
MANAGER_ID = 'manager_id'

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

# sns.set(color_codes=True)
# sns.set(style="whitegrid", color_codes=True)

# pd.set_option('display.max_columns', 500)
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


def do_test(folds):
    df = load_train()
    space= {
        'n_estimators':hp.qnormal('n_estimators', 1000, 200, 10),
        'learning_rate':hp.normal('learning_rate',0.1, 0.05)
    }
    trials = Trials()
    best = fmin(lambda s:simple_cross_val(folds, s, df, trials), space=space, algo=tpe.suggest, trials=trials, max_evals=100)

    print best
    print get_the_best_loss(trials)

def blja_test():
    space= {
        'n_estimators':hp.qnormal('n_estimators', 1000, 200, 10),
        'learning_rate':hp.normal('learning_rate',0.1, 0.05)
    }
    for x in range(1000):
        print pyll.stochastic.sample(space)

#(0.61509489625789615, [0.61124170916042475, 0.61371758902339113, 0.61794752159334343, 0.61555861194203254, 0.61700904957028924])
def simple_cross_val(folds, s, df, trials):
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]

    res = []
    learning_rate = s['learning_rate']
    n_estimators = int(s['n_estimators'])
    print 'n_estimators={}, learning_rate={}'.format(s['n_estimators'], learning_rate)
    if n_estimators<=0 or learning_rate<=0:
        return {'loss':100, 'status': STATUS_FAIL}
    for h in range(folds):
        train_df, test_df = split_df(df, 0.7)

        train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
        del train_df[TARGET]
        del test_df[TARGET]

        train_df = train_df[features]
        test_df = test_df[features]

        train_arr, test_arr = train_df.values, test_df.values

        estimator = xgb.XGBClassifier(n_estimators=n_estimators, objective='multi:softprob', learning_rate=learning_rate)
        # estimator = RandomForestClassifier(n_estimators=1000)
        estimator.fit(train_arr, train_target)

        # plot feature importance
        # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
        # sns.barplot(ffs, [x for x in estimator.feature_importances_])
        # sns.plt.show()


        # print estimator.feature_importances_
        proba = estimator.predict_proba(test_arr)
        l = log_loss(test_target, proba)
        # print l
        res.append(l)

    loss = np.mean(res)
    print 'current_loss={}, best={}'.format(loss, get_the_best_loss(trials))
    print '\n\n'
    return {'loss': loss, 'status': STATUS_OK}


def explore_target():
    df = load_train()[[TARGET]]
    df = pd.get_dummies(df)
    print df.mean()

    # print man_id_cross_val(3)
    # submit_mngr_id()

# print simple_cross_val(5)
do_test(3)
# blja_test()