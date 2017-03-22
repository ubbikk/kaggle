import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import boxcox
from sklearn.metrics import log_loss

src_folder = '/home/dpetrovskyi/PycharmProjects/kaggle/src'
os.chdir(src_folder)
import sys
sys.path.append(src_folder)

TARGET = u'interest_level'
MANAGER_ID = 'manager_id'
BUILDING_ID = 'building_id'
LATITUDE='latitude'
LONGITUDE='longitude'
PRICE='price'
BATHROOMS='bathrooms'
BEDROOMS='bedrooms'
DESCRIPTION='description'
DISPLAY_ADDRESS='display_address'
STREET_ADDRESS='street_address'
LISTING_ID='listing_id'
PRICE_PER_BEDROOM = 'price_per_bedroom'

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

#(0.61509489625789615, [0.61124170916042475, 0.61371758902339113, 0.61794752159334343, 0.61555861194203254, 0.61700904957028924])
def simple_loss(df):
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

    estimator = xgb.XGBClassifier(n_estimators=1000, objective='multi:softprob')
    # estimator = RandomForestClassifier(n_estimators=1000)
    estimator.fit(train_arr, train_target)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()


    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)
    return log_loss(test_target, proba)



def with_per_bedroom_loss(df):
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_month", "created_day", PRICE_PER_BEDROOM]
    mngr_features_only = ['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']


    train_df, test_df = split_df(df, 0.7)
    train_df, test_df = process_price_per_bedroom(train_df, test_df)
    train_df, test_df = process_manager_id(train_df, test_df)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features+mngr_features_only]
    test_df = test_df[features+mngr_features_only]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=1000, objective='multi:softprob')
    # estimator = RandomForestClassifier(n_estimators=1000)
    estimator.fit(train_arr, train_target)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()


    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)
    return log_loss(test_target, proba)

def do_test(num, fp):
    neww=[]
    df = load_train()
    for x in range(num):
        loss = with_per_bedroom_loss(df)
        print loss
        neww.append(loss)

    print np.mean(neww), neww
    with open(fp, 'w+') as f:
        json.dump(neww, f)


def process_price_per_bedroom(train_df, test_df):
    for df in [train_df, test_df]:
        df[PRICE_PER_BEDROOM] = df[BEDROOMS].apply(lambda s: 1 if s==0 else s)
        df[PRICE_PER_BEDROOM]= df[PRICE]/df[BEDROOMS]

    return train_df, test_df

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


do_test(1000, '/home/dpetrovskyi/PycharmProjects/kaggle/trash/mng_and_per_bedroom_with_err.json')
