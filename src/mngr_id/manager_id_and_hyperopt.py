import json
import os
import seaborn as sns
import pandas as pd
from collections import OrderedDict

from hyperopt import STATUS_FAIL
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp, tpe, fmin
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
from v2w import avg_vector_df, load_model, avg_vector_df_and_pca

TARGET = u'interest_level'
MANAGER_ID = 'manager_id'

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

# sns.set(color_codes=True)
# sns.set(style="whitegrid", color_codes=True)

os.chdir('/home/ubik/PycharmProjects/kaggle/src')
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

    return df


def submit_mngr_id():
    train_df = load_train()
    test_df = load_test()

    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]
    features_and_mngr = features + ['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # features_and_mngr_without_target = features + ['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']


    train_df, test_df = process_manager_id(train_df, test_df)

    train_target = train_df[TARGET].values
    # del train_df[TARGET]


    train_arr, test_arr = train_df[features_and_mngr].values, test_df[features_and_mngr].values

    estimator = xgb.XGBClassifier(n_estimators=1000)
    estimator.fit(train_arr, train_target)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()

    proba = estimator.predict_proba(test_arr)
    classes = [x for x in estimator.classes_]
    print classes
    for cl in classes:
        test_df[cl] = proba[:, classes.index(cl)]

    res = test_df[['listing_id', 'high', 'medium', 'low']]
    res.to_csv('/home/dpetrovskyi/PycharmProjects/kaggle/src/results.csv', index=False)


def simple_cross_val(folds):
    df = load_train()
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]

    res = []
    for h in range(folds):
        train_df, test_df = split_df(df, 0.7)

        train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
        del train_df[TARGET]
        del test_df[TARGET]

        train_df = train_df[features]
        test_df = test_df[features]

        train_arr, test_arr = train_df.values, test_df.values

        estimator = xgb.XGBClassifier(n_estimators=1000)
        estimator.fit(train_arr, train_target)

        # plot feature importance
        # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
        # sns.barplot(ffs, [x for x in estimator.feature_importances_])
        # sns.plt.show()


        # print estimator.feature_importances_
        proba = estimator.predict_proba(test_arr)
        l = log_loss(test_target, proba)
        print l
        res.append(l)

    return np.mean(res), res

def get_the_best_loss(trials):
    try:
        return trials.best_trial['result']['loss']
    except:
        return None


def do_test(folds):
    df = load_train()
    space= {
        'n_estimators':hp.qnormal('n_estimators', 1000, 200, 10),
        'learning_rate':hp.normal('learning_rate',0.1, 0.05),
        'gamma':hp.choice('gamma', [0,0.1, 0.01, 0.2]),
        'max_depth':hp.choice('max_depth', [2,3,4,5]),
        'min_child_weight':hp.choice('min_child_weight', [1,2,3])
    }
    trials = Trials()
    best = fmin(lambda s:man_id_cross_val(folds, s, df, trials), space=space, algo=tpe.suggest, trials=trials, max_evals=10000)

    print best
    print get_the_best_loss(trials)

counter = 0

#(0.59205695719975504, [0.58851806059109535, 0.59472482939365567, 0.5929279816145141])
def man_id_cross_val(folds, s, df, trials):
    global counter
    counter+=1
    # df = load_train()
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]
    mngr_features_only = ['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    features_and_mngr = features + mngr_features_only

    res = []
    learning_rate = s['learning_rate']
    gamma = s['gamma']
    n_estimators = int(s['n_estimators'])
    max_depth = int(s['max_depth'])
    min_child_weight = int(s['min_child_weight'])

    print 'n_estimators={}, learning_rate={}, gamma={}, max_depth={}, min_child_weight={}'.format(s['n_estimators'], learning_rate, gamma, max_depth, min_child_weight)
    if n_estimators<=0 or learning_rate<=0:
        return {'loss':100, 'status': STATUS_FAIL}

    for h in range(folds):
        train_df, test_df = split_df(df, 0.7)

        train_df, test_df = process_manager_id(train_df, test_df)

        train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
        del train_df[TARGET]
        del test_df[TARGET]

        train_df = train_df[features_and_mngr]
        test_df = test_df[features_and_mngr]

        train_arr, test_arr = train_df.values, test_df.values

        estimator = xgb.XGBClassifier(
            n_estimators=n_estimators,
            objective='multi:softprob',
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma
        )
        estimator.fit(train_arr, train_target)

        # plot feature importance
        # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
        # sns.barplot(ffs, [x for x in estimator.feature_importances_])
        # sns.plt.show()


        # print estimator.feature_importances_
        proba = estimator.predict_proba(test_arr)
        l = log_loss(test_target, proba)
        res.append(l)

    loss = np.mean(res)
    print '#{}: current_loss={}, best={}'.format(counter, loss, get_the_best_loss(trials))
    print '\n\n'
    return {'loss': loss, 'status': STATUS_OK}


def process_manager_id(train_df, test_df):
    cutoff = 25
    """@type train_df: pd.DataFrame"""
    df = train_df[[MANAGER_ID, TARGET]]
    df = pd.get_dummies(df, columns=[TARGET])
    agg = OrderedDict([
        (MANAGER_ID, {'count': 'count'}),
        ('interest_level_high', {'high': 'mean'}),
        ('interest_level_medium', {'medium': 'mean'}),
        ('interest_level_low', {'low': 'mean'})
    ])
    df = df.groupby(MANAGER_ID).agg(agg)

    df.columns = ['man_id_count', 'man_id_high', 'man_id_medium', 'man_id_low']

    big = df['man_id_count'] >= cutoff
    small = ~big
    bl = pd.get_dummies(train_df, columns=[TARGET])[['interest_level_high', 'interest_level_medium', 'interest_level_low']].mean()#[big]
    df.loc[small, ['man_id_high', 'man_id_medium', 'man_id_low']] = bl.values

    df = df[['man_id_high', 'man_id_medium', 'man_id_low']]
    train_df = pd.merge(train_df, df, left_on=MANAGER_ID, right_index=True)

    test_df = pd.merge(test_df, df, left_on=MANAGER_ID, right_index=True, how='left')
    test_df.loc[test_df['man_id_high'].isnull(), ['man_id_high', 'man_id_medium', 'man_id_low']] = bl.values

    train_df['manager_skill'] = train_df['man_id_high'] * 2 + train_df['man_id_medium']
    test_df['manager_skill'] = test_df['man_id_high'] * 2 + test_df['man_id_medium']

    return train_df, test_df


def explore_target():
    df = load_train()[[TARGET]]
    df = pd.get_dummies(df)
    print df.mean()

    # print man_id_cross_val(3)
    # submit_mngr_id()

# print man_id_cross_val(3)
do_test(5)