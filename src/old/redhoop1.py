import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

TARGET = u'interest_level'
MANAGER_ID = 'manager_id'

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

# sns.set(color_codes=True)
# sns.set(style="whitegrid", color_codes=True)

os.chdir('/home/dpetrovskyi/PycharmProjects/kaggle/src')
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


def man_id_cross_val(folds):
    df = load_train()
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day", TARGET]
    mngr_features_only = ['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    features_and_mngr = features + mngr_features_only

    res = []
    for h in range(folds):
        train_df, test_df = split_df(df, 0.7)

        train_df, test_df = process_manager_id(train_df, test_df)
        train_df = train_df[mngr_features_only + [TARGET]]
        test_df = test_df[mngr_features_only + [TARGET]]

        # train_df = train_df[features]
        # test_df = test_df[features]

        train_target = train_df[TARGET].values
        del train_df[TARGET]
        test_target = test_df[TARGET].values
        del test_df[TARGET]

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
        res.append(l)

    return np.mean(res), res


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
    bl = df[['man_id_high', 'man_id_medium', 'man_id_low']][big].mean()
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

print simple_cross_val(5)
