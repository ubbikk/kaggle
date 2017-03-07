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

src_folder = '/home/dpetrovskyi/PycharmProjects/kaggle/src'
os.chdir(src_folder)
import sys
sys.path.append(src_folder)

from v2w import avg_vector_df, load_model, avg_vector_df_and_pca

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

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

# sns.set(color_codes=True)
# sns.set(style="whitegrid", color_codes=True)

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


def bldng_id_validation(df):
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]

    # bldng_features_only = ['bldng_id_high', 'bldng_id_medium', 'bldng_id_low', 'bldng_skill']
    features_and_bldng = features + ['bldng_id_high', 'bldng_id_medium', 'bldng_id_low', 'bldng_skill']

    res = []
    train_df, test_df = split_df(df, 0.7)

    train_df, test_df = process_building_id(train_df, test_df)
    train_df = train_df[features_and_bldng + [TARGET]]
    test_df = test_df[features_and_bldng + [TARGET]]


    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
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
    return log_loss(test_target, proba)

def most_frequent(l):
    m = {}
    for x in l:
        if x in m:
            m[x]+=1
        else:
            m[x]=1

    m = [(k,v) for k,v in m.iteritems()]
    m.sort(key=lambda s: s[1], reverse=True)
    return m[0][0]

def fill_zero_bid(train_df, test_df):
    #part1
    """@type train_df: pd.DataFrame"""
    """@type test_df: pd.DataFrame"""
    lat_lon = 'lat_lon'
    temp_cols=['name', lat_lon]
    train_df['name'] = 'train'
    test_df['name']='test'
    merged = pd.concat([train_df, test_df])
    merged[lat_lon] = zip(merged[LATITUDE], merged[LONGITUDE])
    bid_zero_df = merged[merged[BUILDING_ID]=='0']
    bid_not_zero_df = merged[merged[BUILDING_ID]!='0']

    bl = bid_not_zero_df.groupby(lat_lon)[BUILDING_ID].apply(most_frequent).to_frame()
    bl = pd.merge(bid_zero_df, bl, left_on=lat_lon, right_index=True, how='left')
    merged.loc[bid_zero_df.index, BUILDING_ID]=bl[BUILDING_ID+'_y']

    #part2
    bid_zero_df = merged[merged[BUILDING_ID].isnull()]
    bid_not_zero_df = merged[~merged[BUILDING_ID].isnull()]
    bid_not_zero_df= bid_not_zero_df[[BUILDING_ID, LATITUDE, LONGITUDE]]

    tree = KDTree(bid_not_zero_df[[LATITUDE, LONGITUDE]].as_matrix())
    bl = bid_zero_df[[LATITUDE, LONGITUDE]]
    bl['zip'] = zip(bl[LATITUDE], bl[LONGITUDE])
    bl['zip'] = bl['zip'].apply(np.array)
    bl['query_res'] = bl['zip'].apply(lambda x: tree.query(x))
    bl['ind']= bl['query_res'].apply(lambda x: x[1])
    bl[BUILDING_ID]= bl['ind'].apply(lambda x: bid_not_zero_df.iloc[x, 0])

    merged.loc[bl.index, BUILDING_ID]=bl[BUILDING_ID]


def test():
    fill_zero_bid(load_train(), load_test())


def process_building_id(train_df, test_df):
    cutoff = 20
    """@type train_df: pd.DataFrame"""
    df = train_df[[BUILDING_ID, TARGET]]
    df = pd.get_dummies(df, columns=[TARGET])
    agg = OrderedDict([
        (BUILDING_ID, {'count': 'count'}),
        ('interest_level_high', {'high': 'mean'}),
        ('interest_level_medium', {'medium': 'mean'}),
        ('interest_level_low', {'low': 'mean'})
    ])
    df = df.groupby(BUILDING_ID).agg(agg)

    df.columns = ['bldng_id_count', 'bldng_id_high', 'bldng_id_medium', 'bldng_id_low']

    big = df['bldng_id_count'] >= cutoff
    small = ~big
    bl = df[['bldng_id_high', 'bldng_id_medium', 'bldng_id_low']][big].mean()
    df.loc[small, ['bldng_id_high', 'bldng_id_medium', 'bldng_id_low']] = bl.values

    df = df[['bldng_id_high', 'bldng_id_medium', 'bldng_id_low']]
    train_df = pd.merge(train_df, df, left_on=BUILDING_ID, right_index=True)

    test_df = pd.merge(test_df, df, left_on=BUILDING_ID, right_index=True, how='left')
    test_df.loc[test_df['bldng_id_high'].isnull(), ['bldng_id_high', 'bldng_id_medium', 'bldng_id_low']] = bl.values

    train_df['bldng_skill'] = train_df['bldng_id_high'] * 2 + train_df['bldng_id_medium']
    test_df['bldng_skill'] = test_df['bldng_id_high'] * 2 + test_df['bldng_id_medium']

    return train_df, test_df

def do_test(num, fp):
    neww=[]
    df = load_train()
    for x in range(num):
        loss = bldng_id_validation(df)
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


# do_test(50, '/home/dpetrovskyi/PycharmProjects/kaggle/trash/building_id_1.json')
test()

