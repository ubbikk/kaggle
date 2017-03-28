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
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.spatial import KDTree
import math
from pymongo import MongoClient



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
F_COL=u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'

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

# train_file = '../data/redhoop/train.json'
# test_file = '../data/redhoop/test.json'



#========================================================
#LISTING_ID

def process_listing_id(train_df, test_df):
    return train_df, test_df, [LISTING_ID]


#========================================================






#========================================================
#MNGR CATEG

def hcc_encode(train_df, test_df, variable, binary_target, k=5, f=1, g=1, r_k = 0.01, folds=5):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    prior_prob = train_df[binary_target].mean()
    hcc_name = "_".join(["hcc", variable, binary_target])

    skf = StratifiedKFold(folds)
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        grouped = big.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
        grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
        grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

        if hcc_name in small.columns:
            del small[hcc_name]
        small = pd.merge(small, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
        small.loc[small[hcc_name].isnull(), hcc_name] = prior_prob
        small[hcc_name]= small[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(small))
        train_df.loc[small.index, hcc_name] = small[hcc_name]

    grouped = train_df.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    test_df = pd.merge(test_df, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
    test_df.loc[test_df[hcc_name].isnull(), hcc_name] = prior_prob

    return train_df, test_df, hcc_name


def get_exp_lambda(k,f):
    def res(n):
        return 1/(1+math.exp(float(k-n)/f))
    return res


def process_mngr_categ_preprocessing(train_df, test_df):
    col = MANAGER_ID
    new_cols=[]
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s=='high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s=='medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols

#========================================================


#========================================================
#BID_CATEG

def designate_single_observations(train_df, test_df, col):
    new_col = '{}_grouped_single_obs'.format(col)
    bl=pd.concat([train_df, test_df]).groupby(col)[col].count()
    bl= bl[bl==1]
    bl = set(bl.index.values)
    train_df[new_col] = train_df[col].apply(lambda s: s if s not in bl else 'single_obs')
    test_df[new_col] = test_df[col].apply(lambda s: s if s not in bl else 'single_obs')
    return train_df, test_df, new_col

def process_bid_categ_preprocessing(train_df, test_df):
    col = BUILDING_ID
    new_cols=[]
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s=='high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s=='medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols

#========================================================

#========================================================
#MANAGER NUM


def process_manager_num(train_df, test_df):
    mngr_num_col = 'manager_num'
    df = train_df.groupby(MANAGER_ID)[MANAGER_ID].count()
    # df[df<=1]=-1
    df = df.apply(float)
    df = df.to_frame(mngr_num_col)
    train_df = pd.merge(train_df, df, left_on=MANAGER_ID, right_index=True)
    test_df = pd.merge(test_df, df, left_on=MANAGER_ID, right_index=True, how='left')

    return train_df, test_df, [mngr_num_col]

#========================================================



#========================================================
#BID NUM


def process_bid_num(train_df, test_df):
    bid_num_col = 'bid_num'
    df = train_df.groupby(BUILDING_ID)[BUILDING_ID].count()
    # df[df<=1]=-1
    df = df.apply(float)
    df = df.to_frame(bid_num_col)
    train_df = pd.merge(train_df, df, left_on=BUILDING_ID, right_index=True)
    test_df = pd.merge(test_df, df, left_on=BUILDING_ID, right_index=True, how='left')

    return train_df, test_df, [bid_num_col]


#========================================================



#========================================================
#TOP 50 GROUPED FEATURES

COL = 'normalized_features'


def normalize_df(df):
    df[COL] = df[F_COL].apply(lambda l: [x.lower() for x in l])


def lambda_in(in_arr):
    def is_in(l):
        for f in l:
            for t in in_arr:
                if t in f:
                    return 1

        return 0

    return is_in


def lambda_equal(val):
    def is_equal(l):
        for f in l:
            if f.strip() == val:
                return 1

        return 0

    return is_equal


def lambda_two_arr(arr1, arr2):
    def is_in(l):
        for f in l:
            for x in arr1:
                for y in arr2:
                    if x in f and y in f:
                        return 1
        return 0

    return is_in


GROUPING_MAP=OrderedDict(
    [('elevator', {'vals': ['elevator'], 'type': 'in'}),
     ('hardwood floors', {'vals': ['hardwood'], 'type': 'in'}),
     ('cats allowed', {'vals': ['cats'], 'type': 'in'}),
     ('dogs allowed', {'vals': ['dogs'], 'type': 'in'}),
     ('doorman', {'vals': ['doorman', 'concierge'], 'type': 'in'}),
     ('dishwasher', {'vals': ['dishwasher'], 'type': 'in'}),
     ('laundry in building', {'vals': ['laundry'], 'type': 'in'}),
     ('no fee', {'vals': ['no fee', 'no broker fee', 'no realtor fee'], 'type': 'in'}),
     ('reduced fee', {'vals': ['reduced fee', 'reduced-fee', 'reducedfee'], 'type': 'in'}),
     ('fitness center', {'vals': ['fitness'], 'type': 'in'}),
     ('pre-war', {'vals': ['pre-war', 'prewar'], 'type': 'in'}),
     ('roof deck', {'vals': ['roof'], 'type': 'in'}),
     ('outdoor space',{'vals': ['outdoor space', 'outdoor-space', 'outdoor areas', 'outdoor entertainment'], 'type': 'in'}),
     ('common outdoor space',{'vals': ['common outdoor', 'publicoutdoor', 'public-outdoor', 'common-outdoor'], 'type': 'in'}),
     ('private outdoor space', {'vals': ['private outdoor', 'private-outdoor', 'privateoutdoor'], 'type': 'in'}),
     ('dining room', {'vals': ['dining'], 'type': 'in'}),
     ('high speed internet', {'vals': ['internet'], 'type': 'in'}),
     ('balcony', {'vals': ['balcony'], 'type': 'in'}),
     ('swimming pool', {'vals': ['swimming', 'pool'], 'type': 'in'}),
     ('new construction', {'vals': ['new construction'], 'type': 'in'}),
     ('terrace', {'vals': ['terrace'], 'type': 'in'}),
     ('exclusive', {'vals': ['exclusive'], 'type': 'equal'}),
     ('loft', {'vals': ['loft'], 'type': 'in'}),
     ('garden/patio', {'vals': ['garden'], 'type': 'in'}),
     ('wheelchair access', {'vals': ['wheelchair'], 'type': 'in'}),
     ('fireplace', {'vals': ['fireplace'], 'type': 'in'}),
     ('simplex', {'vals': ['simplex'], 'type': 'in'}),
     ('lowrise', {'vals': ['lowrise', 'low-rise'], 'type': 'in'}),
     ('garage', {'vals': ['garage'], 'type': 'in'}),
     ('furnished', {'vals': ['furnished'], 'type': 'equal'}),
     ('multi-level', {'vals': ['multi-level', 'multi level', 'multilevel'], 'type': 'in'}),
     ('high ceilings', {'vals': ['high ceilings', 'highceilings', 'high-ceilings'], 'type': 'in'}),
     ('parking space', {'vals': ['parking'], 'type': 'in'}),
     ('live in super', {'vals': ['super'], 'vals2': ['live', 'site'], 'type': 'two'}),
     ('renovated', {'vals': ['renovated'], 'type': 'in'}),
     ('green building', {'vals': ['green building'], 'type': 'in'}),
     ('storage', {'vals': ['storage'], 'type': 'in'}),
     ('washer', {'vals': ['washer'], 'type': 'in'}),
     ('stainless steel appliances', {'vals': ['stainless'], 'type': 'in'})])


def process_features(df):
    normalize_df(df)
    new_cols=[]
    for col, m in GROUPING_MAP.iteritems():
        new_cols.append(col)
        tp = m['type']
        if tp == 'in':
            df[col] = df[COL].apply(lambda_in(m['vals']))
        elif tp=='equal':
            df[col] = df[COL].apply(lambda_equal(m['vals'][0]))
        elif tp=='two':
            df[col] = df[COL].apply(lambda_two_arr(m['vals'], m['vals2']))
        else:
            raise Exception()

    return df, new_cols


#========================================================



#========================================================
#WRITTING RESULTS


def out(l, loss, l_1K, loss1K, num, t):
    print '\n\n'
    print '#{}'.format(num)
    if loss1K is not None:
        print 'loss1K {}'.format(loss1K)
        print 'avg_loss1K {}'.format(np.mean(l_1K))
        print get_3s_confidence_for_mean(l_1K)
        print

    print 'loss {}'.format(loss)
    print 'avg_loss {}'.format(np.mean(l))
    print get_3s_confidence_for_mean(l)
    print 'std {}'.format(np.std(l))
    print 'time {}'.format(t)

def get_3s_confidence_for_mean(l):
    std = np.std(l)/math.sqrt(len(l))
    m = np.mean(l)
    start = m -3*std
    end = m+3*std

    return '3s_confidence: [{}, {}]'.format(start, end)

def write_results(l, name, ii, fldr=None):
    client = MongoClient('10.20.0.144', 27017)
    db = client.renthop_results
    collection = db[name]
    losses = l[len(l)-1]
    importance = ii[len(ii)-1]
    collection.insert_one({'results':losses, 'importance':importance})
    fp = name+'.json' if fldr is None else os.path.join(name+'.json')
    ii_fp = name+'_importance.json' if fldr is None else os.path.join(name+'_importance.json')
    with open(fp, 'w+') as f:
        json.dump(l, f)
    with open(ii_fp, 'w+') as f:
        json.dump(ii, f)

#========================================================



#========================================================
#VALIDATION
def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]


def load_train():
    return basic_preprocess(pd.read_json(train_file))


def load_test():
    return basic_preprocess(pd.read_json(test_file))

def process_outliers_lat_long(train_df, test_df):
    min_lat=40
    max_lat=41
    min_long=-74.1
    max_long=-73

    good_lat = (train_df[LATITUDE] < max_lat) & (train_df[LATITUDE] > min_lat)
    good_long = (train_df[LONGITUDE] < max_long) & (train_df[LONGITUDE] > min_long)

    train_df = train_df[good_lat & good_long]

    bed_lat = (test_df[LATITUDE] >=max_lat) | (test_df[LATITUDE] <=min_lat)
    bed_long = (test_df[LONGITUDE] >= max_long) | (test_df[LONGITUDE] <= min_long)
    test_df[LATITUDE][bed_lat] = train_df[LATITUDE].mean()
    test_df[LONGITUDE][bed_long]=train_df[LONGITUDE].mean()

    return train_df, test_df


def basic_preprocess(df):
    df['num_features'] = df[u'features'].apply(len)
    df['num_photos'] = df['photos'].apply(len)
    df['word_num_in_descr'] = df['description'].apply(lambda x: len(x.split(' ')))
    df["created"] = pd.to_datetime(df["created"])
    # df["created_year"] = df["created"].dt.year
    df[CREATED_MONTH] = df["created"].dt.month
    df[CREATED_DAY] = df["created"].dt.day
    df[CREATED_HOUR] = df["created"].dt.hour
    df[CREATED_MINUTE] = df["created"].dt.minute
    df[DAY_OF_WEEK] = df['created'].dt.dayofweek
    bc_price, tmp = boxcox(df['price'])
    df['bc_price'] = bc_price

    return df


def get_loss_at1K(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    return results_on_test[1000]

def loss_with_per_tree_stats(df, new_cols):
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_month", "created_day", CREATED_HOUR, CREATED_MINUTE, DAY_OF_WEEK]
    features+=new_cols

    train_df, test_df = split_df(df, 0.7)

    train_df, test_df, new_cols = process_mngr_categ_preprocessing(train_df, test_df)
    features+=new_cols

    train_df, test_df, new_cols = process_manager_num(train_df, test_df)
    features+=new_cols

    train_df, test_df, new_cols = process_bid_categ_preprocessing(train_df, test_df)
    features+=new_cols

    train_df, test_df, new_cols = process_bid_num(train_df, test_df)
    features+=new_cols

    train_df, test_df, new_cols = process_listing_id(train_df, test_df)
    features+=new_cols

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values
    print features

    estimator = xgb.XGBClassifier(n_estimators=1100, objective='mlogloss')
    eval_set = [(train_arr, train_target), (test_arr, test_target)]
    estimator.fit(train_arr, train_target, eval_set=eval_set, eval_metric='mlogloss', verbose=False)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()


    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)

    loss = log_loss(test_target, proba)
    loss1K = get_loss_at1K(estimator)
    return loss, loss1K, xgboost_per_tree_results(estimator), estimator.feature_importances_

def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    results_on_train = estimator.evals_result()['validation_0']['mlogloss']
    return {
        'train':results_on_train,
        'test':results_on_test
    }

def do_test_with_xgboost_stats_per_tree(num, fp):
    l = []
    results =[]
    l_1K=[]
    train_df = load_train()
    train_df, new_cols = process_features(train_df)
    ii=[]
    for x in range(num):
        t=time()
        df=train_df.copy()

        loss, loss1K, res , imp= loss_with_per_tree_stats(df, new_cols)
        ii.append(imp.tolist())

        t=time()-t
        l.append(loss)
        l_1K.append(loss1K)
        results.append(res)

        out(l, loss, l_1K, loss1K, x, t)
        write_results(results, fp, ii)


do_test_with_xgboost_stats_per_tree(1000, 'all_and_fixed_hcc')
