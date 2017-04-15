import json
import os
import traceback
from time import time, sleep

import seaborn as sns
import pandas as pd
from collections import OrderedDict

import sys
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
F_COL = u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE = 'created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
CREATED = 'created'
LABEL = 'lbl'
BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'
COL = 'normalized_features'
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'
NEI = 'neighbourhood'
BORO = 'boro'

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
train_geo_file = '../../data/redhoop/with_geo/train_geo.json'
test_geo_file = '../../data/redhoop/with_geo/test_geo.json'
rent_file = '../with_geo/data/neis_from_renthop_lower.json'
seeds_fp = '../../seeds.json'

# train_file = '../data/redhoop/train.json'
# test_file = '../data/redhoop/test.json'
# train_geo_file = '../data/redhoop/with_geo/train_geo.json'
# test_geo_file = '../data/redhoop/with_geo/test_geo.json'
# rent_file = 'with_geo/data/neis_from_renthop_lower.json'
# seeds_fp = '../seeds.json'


#########################################################################################
# Mongo Control
#########################################################################################

SEEDS = json.load(open(seeds_fp))


def getN(mongo_host, name, experiment_max_time):
    client = MongoClient(mongo_host, 27017)
    db = client[name]
    collection = db['splits_control'.format(name)]
    res = [x for x in collection.find()]

    res.sort(key=lambda s: s['N'])

    for con in res:
        if (not con['finished']) and (time()-con['time'] > experiment_max_time):
            N = con['N']
            collection.replace_one({'N': N}, {'N': N, 'time': time(), 'finished': False})
            return N

    N = len(res)
    collection.insert_one({'N': N, 'time': time(), 'finished': False})

    return N


def split_from_N(df, N):
    folds = 5
    seed = SEEDS[N / folds]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    gen = skf.split(np.zeros(len(df)), df['interest_level'])
    kfold_ind = N % folds

    counter = 0
    for big_ind, small_ind in gen:
        if counter == kfold_ind:
            return df.iloc[big_ind], df.iloc[small_ind]
        counter += 1


def get_next_split(df, mongo_host, name, experiment_max_time):
    N = getN(mongo_host, name, experiment_max_time)
    train_df, test_df = split_from_N(df, N)

    client = MongoClient(mongo_host, 27017)
    db = client[name]

    collection = db['splits']
    collection.insert_one({'N': N, 'train': list(train_df.index.values), 'test': list(test_df.index.values)})

    return train_df, test_df


def complete_split_mongo(N, name, mongo_host, probs, test_indexes, losses, importance, f_names):
    client = MongoClient(mongo_host, 27017)
    db = client[name]

    collection = db['probs']
    collection.insert_one({'N': N, 'val': probs, 'index':test_indexes})

    collection = db['losses']
    collection.insert_one({'N': N, 'val': losses})

    collection = db['importance']
    collection.insert_one({'N': N, 'val': importance})

    collection = db['features']
    collection.insert_one({'N': N, 'val': f_names})

    collection = db['splits_control'.format(name)]
    collection.replace_one({'N': N}, {'N': N, 'time': time(), 'finished': True})


def get_probs_from_est(estimator, proba, test_df):
    classes = [x for x in estimator.classes_]
    res = {}
    for cl in classes:
        p=proba[:, classes.index(cl)]
        res[cl] = [a.item() for a in p]
    return res, [x for x in test_df.index.values]


def complete_split_file(ii, l, name):
    fp = name + '_results.json'
    ii_fp = name + '_importance.json'
    with open(fp, 'w+') as f:
        json.dump(l, f)
    with open(ii_fp, 'w+') as f:
        json.dump(ii, f)


#########################################################################################
# Mongo Control
#########################################################################################


#########################################################################################
# Writing Results
#########################################################################################

def write_results(N, name, mongo_host, probs, test_indexes, l_results, ii_importance, f_names):
    losses = l_results[len(l_results) - 1]
    importance = ii_importance[len(ii_importance) - 1]

    retries = 5
    while retries >= 0:
        try:
            complete_split_mongo(N, name, mongo_host, probs, test_indexes,  losses, importance, f_names)
            break
        except:
            traceback.print_exc()
            retries -= 1
            sleep(30)

    try:
        complete_split_file(ii_importance, l_results, name)
    except:
        traceback.print_exc()


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
    std = np.std(l) / math.sqrt(len(l))
    m = np.mean(l)
    start = m - 3 * std
    end = m + 3 * std

    return '3s_confidence: [{}, {}]'.format(start, end)


#########################################################################################
# Writing Results
#########################################################################################




#########################################################################################
# loading data
#########################################################################################


def load_df(file, geo_file):
    df = pd.read_json(file)
    geo = pd.read_json(geo_file)
    df[NEI] = geo[NEI]
    df['tmp'] = df[NEI].apply(transform_geo_to_rent)
    df[NEI_1] = df['tmp'].apply(lambda s: None if s is None else s[0])
    df[NEI_2] = df['tmp'].apply(lambda s: None if s is None else s[1])
    df[NEI_3] = df['tmp'].apply(lambda s: None if s is None else s[2])
    normalize_bed_bath(df)
    return basic_preprocess(df)


def load_train():
    df = load_df(train_file, train_geo_file)
    df[LABEL] = 'train'
    return df


def load_test():
    df = load_df(test_file, test_geo_file)
    df[LABEL] = 'test'
    return df


def load_rent():
    m = json.load(open(rent_file))
    res = {}
    for boro, boro_m in m.iteritems():
        for sub_boro, neis in boro_m.iteritems():
            for n in neis:
                res[n] = [n, sub_boro, boro]

    return res


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


#########################################################################################
# loading data
#########################################################################################

#########################################################################################
# Creating Neis
#########################################################################################
def normalize_bed_bath(df):
    df[BED_NORMALIZED] = df[BEDROOMS].apply(lambda s: s if s <= 3 else 3)

    def norm_bath(s):
        s = round(s)
        if s == 0:
            return 1
        if s >= 2:
            return 2
        return s

    df[BATH_NORMALIZED] = df[BATHROOMS].apply(norm_bath)


EXACT_MAP = {
    'gramercy': 'gramercy park',
    'clinton': "hell's kitchen",
    'turtle bay': 'midtown east',
    'tudor city': 'midtown east',
    'sutton place': 'midtown east',
    'hamilton heights': 'west harlem',
    'bedford stuyvesant': 'bedford-stuyvesant',
    'hunters point': 'long island city',
    'battery park': 'battery park city',
    'manhattanville': 'west harlem',
    'carnegie hill': 'upper east side',
    'stuyvesant town': 'stuyvesant town - peter cooper village',
    'downtown': 'downtown brooklyn',
    'morningside heights': 'west harlem',
    'spuyten duyvil': 'riverdale',
    'prospect lefferts gardens': 'flatbush',
    'greenwood': 'greenwood heights',
    'fort hamilton': 'bay ridge',
    'high bridge': 'highbridge',
    'columbia street waterfront district': 'carroll gardens',
    'ocean parkway': 'midwood',
    'north riverdale': 'riverdale',
    'astoria heights': 'astoria',
    'tremont': 'mount hope',
    'homecrest': 'sheepshead bay',
    'new utrecht': 'borough park',
    'fieldston': 'riverdale',
    'georgetown': 'upper east side',
    'tottenville': 'washington heights',
    'hillcrest': 'kew gardens hills',
    'oakland gardens': 'forest hills',
    'pomonok': 'washington heights',
    'wingate': 'east flatbush',
    'fordham': 'fordham manor',
    'forest hills gardens': 'forest hills',
    'columbus circle': "hell's kitchen"
}

SPECIAL = {
    'midtown': ('midtown', 'midtown manhattan', 'manhattan'),
    'harlem': ('harlem', 'upper manhattan', 'manhattan')
}

ONLY_SECOND = {
    'castle hill': ('2nd', 'east bronx', 'bronx'),
    'throggs neck': ('2nd', 'east bronx', 'bronx'),
    'soundview': ('2nd', 'east bronx', 'bronx'),
    'port morris': ('2nd', 'east bronx', 'bronx'),
}

ONLY_THIRD = {
    'queens village': ('3rd', '3rd', 'queens'),
    'laurelton': ('3rd', '3rd', 'queens')
}


def transform_geo_to_rent(s):
    if s is None:
        return s
    s = s.lower()
    rent = load_rent()
    if s in rent:
        return rent[s]

    if s in EXACT_MAP:
        return rent[EXACT_MAP[s]]

    if s in SPECIAL:
        return SPECIAL[s]

    return ('not_mapped_yet', 'not_mapped_yet', 'not_mapped_yet')


#########################################################################################
# Creating Neis
#########################################################################################




def get_loss_at1K(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    return results_on_test[1000]


def loss_with_per_tree_stats(train_df, test_df, new_cols):
    features, test_df, train_df = process_split(train_df, test_df, new_cols)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values

    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values
    print features

    estimator = xgb.XGBClassifier(n_estimators=1100, objective='mlogloss', subsample=0.8, colsample_bytree=0.8)
    eval_set = [(train_arr, train_target), (test_arr, test_target)]
    estimator.fit(train_arr, train_target, eval_set=eval_set, eval_metric='mlogloss', verbose=False)

    proba = estimator.predict_proba(test_arr)

    loss = log_loss(test_target, proba)
    loss1K = get_loss_at1K(estimator)
    return loss, loss1K, xgboost_per_tree_results(estimator), \
           estimator.feature_importances_, get_probs_from_est(estimator, proba, test_df), features


def process_split(train_df, test_df, new_cols):
    features = []
    features += new_cols

    # train_df, test_df, new_cols = process_mngr_categ_preprocessing(train_df, test_df)
    # train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    # features += new_cols
    #
    # train_df, test_df, new_cols = process_manager_num(train_df, test_df)
    # train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    # features += new_cols

    return features, test_df, train_df


def process_all_name(train_df, test_df):
    features = ['bathrooms', 'bedrooms', 'latitude',
                'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_month", "created_day",
                CREATED_HOUR, CREATED_MINUTE, DAY_OF_WEEK]

    # train_df, new_cols = process_features(train_df)
    # train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    # features += new_cols
    #
    # train_df, test_df, new_cols = process_mngr_avg_median_price(train_df, test_df)
    # train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    # features += new_cols

    return train_df, test_df, features


def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    results_on_train = estimator.evals_result()['validation_0']['mlogloss']
    return {
        'train': results_on_train,
        'test': results_on_test
    }


def do_test_xgboost(name, mongo_host, experiment_max_time=30*60):
    all_losses = []
    l_results_per_tree = []
    losses_at_1K = []

    train_df = load_train().head(1000)
    test_df = load_test().head(1000)

    train_df, test_df, features = process_all_name(train_df, test_df)

    ii_importance = []
    for counter in range(1000):
        cur_time = time()
        N = getN(mongo_host, name, experiment_max_time)
        train, test = split_from_N(train_df.copy(), N)

        loss, loss1K, losses_per_tree, importance, probs_data, f_names = \
            loss_with_per_tree_stats(train, test, features)
        probs, test_indexes = probs_data

        ii_importance.append(importance.tolist())
        cur_time = time() - cur_time
        all_losses.append(loss)
        losses_at_1K.append(loss1K)
        l_results_per_tree.append(losses_per_tree)

        out(all_losses, loss, losses_at_1K, loss1K, counter, cur_time)
        write_results(N, name, mongo_host, probs,test_indexes, l_results_per_tree, ii_importance, f_names)


name = 'test_processing'
mongo_host='10.20.0.144'

do_test_xgboost(name, mongo_host, 700)