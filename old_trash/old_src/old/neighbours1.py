import json
import os
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

# src_folder = '/home/ubik/PycharmProjects/kaggle/src'
# os.chdir(src_folder)
# import sys
#
# sys.path.append(src_folder)

from v2w import avg_vector_df, load_model, avg_vector_df_and_pca

TARGET = u'interest_level'
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


# def process_neighbours_density(train_df, test_df):
#     r=0.001
#     new_col = 'num_in_distance_{}'.format(r)
#     merged = pd.concat([train_df, test_df])
#     df = merged[[LATITUDE, LONGITUDE]]
#     tree = KDTree(df.values)
#     merged[new_col]=tree.query_ball_point(df.values, r=r)
#     merged[new_col]= merged[new_col].apply(len)
#
#     return merged.loc[train_df.index,:], merged.loc[test_df.index, :]

def process_neighbours_density_merged(merged, r):
    new_col = 'num_in_distance_{}'.format(r)
    df = merged[[LATITUDE, LONGITUDE]]
    tree = KDTree(df.values)
    merged[new_col]=tree.query_ball_point(df.values, r=r)
    merged[new_col]= merged[new_col].apply(len)

    return merged

def explore_neighbours(train_df, test_df):
    r=0.001
    df = train_df[[LATITUDE, LONGITUDE, TARGET]]
    df = pd.get_dummies(df, columns=[TARGET])
    tree = KDTree(df[[LATITUDE, LONGITUDE]].values)
    t=test_df[[LATITUDE, LONGITUDE]]
    t['res']=tree.query_ball_point(t.values, r=r)


# (0.61509489625789615, [0.61124170916042475, 0.61371758902339113, 0.61794752159334343, 0.61555861194203254, 0.61700904957028924])
def simple_loss(df, r):
    density_feature = 'num_in_distance_{}'.format(r)
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day", density_feature]

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


def do_test(num, fp,r):
    neww = []
    df = load_train()
    df = process_neighbours_density_merged(df, r)
    for x in range(num):
        loss = simple_loss(df, r)
        print loss
        neww.append(loss)

    print '\n\n\n\n'
    print 'avg = {}'.format(np.mean(neww))
    with open(fp, 'w+') as f:
        json.dump(neww, f)


def explore_target():
    df = load_train()[[TARGET]]
    df = pd.get_dummies(df)
    print df.mean()


# train_df, test_df = load_train(), load_test()
for r in (0.0001, 0.0003, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
    fp = '/home/ubik/PycharmProjects/kaggle/trash/density_nv_{}.json'.format(r)
    print r
    print fp
    do_test(200, fp, r)
