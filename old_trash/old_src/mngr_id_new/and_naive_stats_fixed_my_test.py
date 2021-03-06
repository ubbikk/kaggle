import json
import os
from time import time

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
CREATED='created'
LABEL='lbl'

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



# ========================================================
# LISTING_ID

def process_listing_id(train_df, test_df):
    return train_df, test_df, [LISTING_ID]


# ========================================================






# ========================================================
# MNGR CATEG

def hcc_encode(train_df, test_df, variable, binary_target, k=5, f=1, g=1, r_k=0.01, folds=5):
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
        small[hcc_name] = small[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(small))
        train_df.loc[small.index, hcc_name] = small[hcc_name]

    grouped = train_df.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    test_df = pd.merge(test_df, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
    test_df.loc[test_df[hcc_name].isnull(), hcc_name] = prior_prob

    return train_df, test_df, hcc_name


def get_exp_lambda(k, f):
    def res(n):
        return 1 / (1 + math.exp(float(k - n) / f))

    return res


def process_mngr_categ_preprocessing(train_df, test_df):
    col = MANAGER_ID
    new_cols = []
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols


# ========================================================


# ========================================================
# BID_CATEG

def designate_single_observations(train_df, test_df, col):
    new_col = '{}_grouped_single_obs'.format(col)
    bl = pd.concat([train_df, test_df]).groupby(col)[col].count()
    bl = bl[bl == 1]
    bl = set(bl.index.values)
    train_df[new_col] = train_df[col].apply(lambda s: s if s not in bl else 'single_obs')
    test_df[new_col] = test_df[col].apply(lambda s: s if s not in bl else 'single_obs')
    return train_df, test_df, new_col


def process_bid_categ_preprocessing(train_df, test_df):
    col = BUILDING_ID
    new_cols = []
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols


# ========================================================

# ========================================================
# MANAGER NUM


def process_manager_num(train_df, test_df):
    mngr_num_col = 'manager_num'
    df = train_df.groupby(MANAGER_ID)[MANAGER_ID].count()
    # df[df<=1]=-1
    df = df.apply(float)
    df = df.to_frame(mngr_num_col)
    train_df = pd.merge(train_df, df, left_on=MANAGER_ID, right_index=True)
    test_df = pd.merge(test_df, df, left_on=MANAGER_ID, right_index=True, how='left')

    return train_df, test_df, [mngr_num_col]


# ========================================================



# ========================================================
# BID NUM


def process_bid_num(train_df, test_df):
    bid_num_col = 'bid_num'
    df = train_df.groupby(BUILDING_ID)[BUILDING_ID].count()
    # df[df<=1]=-1
    df = df.apply(float)
    df = df.to_frame(bid_num_col)
    train_df = pd.merge(train_df, df, left_on=BUILDING_ID, right_index=True)
    test_df = pd.merge(test_df, df, left_on=BUILDING_ID, right_index=True, how='left')

    return train_df, test_df, [bid_num_col]


# ========================================================



# ========================================================
# TOP 50 GROUPED FEATURES

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


GROUPING_MAP = OrderedDict(
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
     ('outdoor space',
      {'vals': ['outdoor space', 'outdoor-space', 'outdoor areas', 'outdoor entertainment'], 'type': 'in'}),
     ('common outdoor space',
      {'vals': ['common outdoor', 'publicoutdoor', 'public-outdoor', 'common-outdoor'], 'type': 'in'}),
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
    new_cols = []
    for col, m in GROUPING_MAP.iteritems():
        new_cols.append(col)
        tp = m['type']
        if tp == 'in':
            df[col] = df[COL].apply(lambda_in(m['vals']))
        elif tp == 'equal':
            df[col] = df[COL].apply(lambda_equal(m['vals'][0]))
        elif tp == 'two':
            df[col] = df[COL].apply(lambda_two_arr(m['vals'], m['vals2']))
        else:
            raise Exception()

    return df, new_cols


# ========================================================


# ========================================================

train_geo_file = '../../data/redhoop/with_geo/train_geo.json'
test_geo_file = '../../data/redhoop/with_geo/test_geo.json'


def load_df(file, geo_file):
    df = pd.read_json(file)
    geo = pd.read_json(geo_file)
    df[NEI] = geo[NEI]
    df['tmp'] = df[NEI].apply(transform_geo_to_rent)
    df[NEI_1] = df['tmp'].apply(lambda s: None if s is None else s[0])
    df[NEI_2] = df['tmp'].apply(lambda s: None if s is None else s[1])
    df[NEI_3] = df['tmp'].apply(lambda s: None if s is None else s[2])
    return basic_preprocess(df)


def load_train():
    df = load_df(train_file, train_geo_file)
    df[LABEL] = 'train'
    return df


def load_test():
    df = load_df(test_file, test_geo_file)
    df[LABEL] = 'test'
    return df


BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'

NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'
NEI = 'neighbourhood'
BORO = 'boro'

rent_file = '../with_geo/data/neis_from_renthop_lower.json'

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


def load_rent():
    m = json.load(open(rent_file))
    res = {}
    for boro, boro_m in m.iteritems():
        for sub_boro, neis in boro_m.iteritems():
            for n in neis:
                res[n] = [n, sub_boro, boro]

    return res


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


def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)


def get_dummy_cols(col_name, col_values):
    return ['{}_{}'.format(col_name, val) for val in col_values]


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


# NEI123
def process_nei123(train_df, test_df):
    df = pd.concat([train_df, test_df])
    normalize_bed_bath(df)
    sz = float(len(df))
    # neis_cols = [NEI_1, NEI_2, NEI_3]
    new_cols = []
    for col in [NEI_1, NEI_2]:
        new_col = 'freq_of_{}'.format(col)
        df[new_col] = df.groupby(col)[PRICE].transform('count')
        df[new_col] = df[new_col] / sz
        new_cols.append(new_col)

    beds_vals = [0, 1, 2, 3]
    for col in [NEI_1, NEI_2, NEI_3]:
        for bed in beds_vals:
            new_col = 'freq_of_{}, with bed={}'.format(col, bed)
            df[new_col] = df.groupby([col, BED_NORMALIZED])[PRICE].transform('count')
            df[new_col] = df[new_col] / sz
            new_cols.append(new_col)

    for col in [NEI_1, NEI_2]:
        new_col = 'median_ratio_of_{}'.format(col)
        df['tmp'] = df.groupby([col, BEDROOMS])[PRICE].transform('median')
        df[new_col] = df[PRICE] - df['tmp']
        df[new_col] = df[new_col] / df['tmp']
        new_cols.append(new_col)

    for col in [NEI_1, NEI_2, NEI_3]:
        vals = set(df[col])
        if None in vals:
            vals.remove(None)
        df = pd.get_dummies(df, columns=[col])
        dummies = get_dummy_cols(col, vals)
        new_cols += dummies

    for d in [train_df, test_df]:
        for col in new_cols:
            d[col] = df.loc[d.index, col]

    return train_df, test_df, new_cols


# ========================================================



# ========================================================
# WRITTING RESULTS


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


def write_results(l, ii, name, mongo_host, fldr=None):  # results, ii, fp, mongo_host
    client = MongoClient(mongo_host, 27017)
    db = client.renthop_results
    collection = db[name]
    losses = l[len(l) - 1]
    importance = ii[len(ii) - 1]
    collection.insert_one({'results': losses, 'importance': importance})
    fp = name + '.json' if fldr is None else os.path.join(name + '.json')
    ii_fp = name + '_importance.json' if fldr is None else os.path.join(name + '_importance.json')
    with open(fp, 'w+') as f:
        json.dump(l, f)
    with open(ii_fp, 'w+') as f:
        json.dump(ii, f)


# ========================================================

def process_mngr_target_ratios_fixed_my(train_df, test_df):
    col = MANAGER_ID
    return process_target_ratios_my(test_df, train_df, col)


def process_target_ratios_my(test_df, train_df, col):
    target_vals = ['high', 'medium', 'low']
    dummies = {k: 'interest_level_{}'.format(k) for k in target_vals}
    new_cols = ['{}_ratio_of_{}'.format(col, t) for t in target_vals]
    df = pd.get_dummies(train_df, columns=[TARGET])
    bl = df[[dummies[t] for t in target_vals] + [col]].groupby(col).sum()
    bl['sum'] = sum([bl[dummies[t]] for t in target_vals])
    bl = bl.rename(columns={'interest_level_{}'.format(t): '{}_ratio_of_{}'.format(col, t) for t in target_vals})
    df = pd.merge(df, bl, left_on=col, right_index=True)
    ratios_part = df[new_cols]
    dumies_part = df[[dummies[t] for t in target_vals]].rename(
        columns={'interest_level_{}'.format(t): '{}_ratio_of_{}'.format(col, t) for t in target_vals})
    df.loc[:, new_cols] = ratios_part - dumies_part
    train_df[new_cols] = df[new_cols]
    train_df['sum_bl'] = df['sum']
    train_df['sum_bl'] -= 1
    for t in target_vals:
        train_df['{}_ratio_of_{}'.format(col, t)] = train_df['{}_ratio_of_{}'.format(col, t)] / train_df['sum_bl']
    for t in target_vals:
        bl['{}_ratio_of_{}'.format(col, t)] = bl['{}_ratio_of_{}'.format(col, t)] / bl['sum']
    bl = bl[new_cols]
    test_df = pd.merge(test_df, bl, left_on=col, right_index=True)

    return train_df, test_df, new_cols

# ========================================================
# VALIDATION
def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]


def shuffle_df(df):
    return df.iloc[np.random.permutation(len(df))]


# def load_train():
#     return basic_preprocess(pd.read_json(train_file))
#
#
# def load_test():
#     return basic_preprocess(pd.read_json(test_file))

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
    features += new_cols

    train_df, test_df = split_df(df, 0.7)

    # train_df, test_df, new_cols = process_mngr_categ_preprocessing(train_df, test_df)
    # train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    # features += new_cols

    train_df, test_df, new_cols = process_manager_num(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_bid_categ_preprocessing(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_bid_num(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_listing_id(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_nei123(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_mngr_target_ratios_fixed_my(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

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
        'train': results_on_train,
        'test': results_on_test
    }


def do_test_with_xgboost_stats_per_tree(num, fp, mongo_host):
    l = []
    results = []
    l_1K = []
    train_df = load_train()
    train_df, new_cols = process_features(train_df)
    ii = []
    for x in range(num):
        t = time()
        df = train_df.copy()

        loss, loss1K, res, imp = loss_with_per_tree_stats(df, new_cols)
        ii.append(imp.tolist())

        t = time() - t
        l.append(loss)
        l_1K.append(loss1K)
        results.append(res)

        out(l, loss, l_1K, loss1K, x, t)
        write_results(results, ii, fp, mongo_host)


do_test_with_xgboost_stats_per_tree(1000, 'and_naive_stats_fixed_my_test', sys.argv[1])

"""
features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'num_features', 'num_photos', 'word_num_in_descr', 'created_month',
'created_day', 'created_hour', 'created_minute', 'dayOfWeek', 'elevator', 'hardwood floors', 'cats allowed', 'dogs allowed',
'doorman', 'dishwasher', 'laundry in building', 'no fee', 'reduced fee', 'fitness center', 'pre-war', 'roof deck', 'outdoor space',
'common outdoor space', 'private outdoor space', 'dining room', 'high speed internet', 'balcony', 'swimming pool', 'new construction',
 'terrace', 'exclusive', 'loft', 'garden/patio', 'wheelchair access', 'fireplace', 'simplex', 'lowrise', 'garage', 'furnished',
  'multi-level', 'high ceilings', 'parking space', 'live in super', 'renovated', 'green building', 'storage', 'washer',
  'stainless steel appliances', 'hcc_manager_id_target_high', 'hcc_manager_id_target_medium', 'manager_num',
  'hcc_building_id_target_high', 'hcc_building_id_target_medium', 'bid_num', 'listing_id', 'freq_of_nei1',
   'freq_of_nei2', 'freq_of_nei1, with bed=0', 'freq_of_nei1, with bed=1', 'freq_of_nei1, with bed=2',
    'freq_of_nei1, with bed=3', 'freq_of_nei2, with bed=0', 'freq_of_nei2, with bed=1', 'freq_of_nei2, with bed=2',
    'freq_of_nei2, with bed=3', 'freq_of_nei3, with bed=0', 'freq_of_nei3, with bed=1', 'freq_of_nei3, with bed=2',
    'freq_of_nei3, with bed=3', 'median_ratio_of_nei1', 'median_ratio_of_nei2', 'nei1_central park', 'nei1_downtown brooklyn',
     'nei1_dyker heights', 'nei1_little italy', 'nei1_long island city', 'nei1_murray hill', 'nei1_rockaway beach', 'nei1_concourse',
     'nei1_richmond hill', 'nei1_bedford-stuyvesant', 'nei1_midtown east', 'nei1_university heights', 'nei1_woodside', 'nei1_coney island',
      'nei1_kew gardens hills', 'nei1_gravesend', 'nei1_red hook', 'nei1_forest hills', 'nei1_marble hill', 'nei1_wakefield', 'nei1_williamsbridge',
       'nei1_hollis', 'nei1_east village', 'nei1_glendale', 'nei1_morris heights', 'nei1_glen oaks', 'nei1_bensonhurst', 'nei1_bushwick', 'nei1_jamaica',
       'nei1_financial district', 'nei1_flatiron district', 'nei1_middle village', 'nei1_prospect heights', 'nei1_greenpoint', 'nei1_brighton beach',
        'nei1_jackson heights', 'nei1_kensington', 'nei1_flushing', 'nei1_inwood', 'nei1_noho', 'nei1_jamaica estates', 'nei1_kingsbridge', 'nei1_boerum hill',
        'nei1_greenwood heights', 'nei1_canarsie', 'nei1_upper east side', 'nei1_flatlands', 'nei1_whitestone', 'nei1_brooklyn heights',
        'nei1_stuyvesant town - peter cooper village', 'nei1_borough park', 'nei1_sheepshead bay', 'nei1_west harlem', 'nei1_south ozone park',
         'nei1_hunts point', 'nei1_brownsville', 'nei1_park slope', 'nei1_highbridge', 'nei1_windsor terrace', 'nei1_roosevelt island',
          'nei1_east harlem', 'nei1_rego park', 'nei1_bedford park', 'nei1_ridgewood', 'nei1_east tremont', 'nei1_cobble hill',
           'nei1_unionport', 'nei1_far rockaway', 'nei1_ozone park', 'nei1_belmont', 'nei1_astoria', 'nei1_fordham manor',
           'nei1_briarwood', 'nei1_gowanus', 'nei1_parkchester', 'nei1_lower east side', 'nei1_mott haven', 'nei1_norwood',
           'nei1_tribeca', 'nei1_chinatown', 'nei1_midtown', 'nei1_clinton hill', 'nei1_chelsea', 'nei1_marine park',
           'nei1_morris park', 'nei1_van cortlandt park', 'nei1_sunset park', 'nei1_garment district', 'nei1_not_mapped_yet',
           'nei1_midwood', 'nei1_bath beach', 'nei1_maspeth', 'nei1_bay ridge', 'nei1_bayside', 'nei1_gramercy park',
           'nei1_sunnyside', 'nei1_carroll gardens', 'nei1_williamsburg', 'nei1_mount hope', 'nei1_pelham bay',
           'nei1_battery park city', 'nei1_west village', 'nei1_flatbush', 'nei1_ocean hill', 'nei1_upper west side',
           "nei1_hell's kitchen", 'nei1_dumbo', 'nei1_east flatbush', 'nei1_washington heights', 'nei1_east elmhurst',
           'nei1_harlem', 'nei1_greenwich village', 'nei1_crown heights', 'nei1_fort greene', 'nei1_corona',
           'nei1_east new york', 'nei1_soho', 'nei1_elmhurst', 'nei1_riverdale', 'nei1_woodhaven', 'nei1_kew gardens',
            'nei2_west bronx', 'nei2_northeastern queens', 'nei2_northwestern brooklyn', 'nei2_southeastern brooklyn',
            'nei2_northwestern queens', 'nei2_eastern brooklyn', 'nei2_central brooklyn', 'nei2_not_mapped_yet',
             'nei2_south brooklyn', 'nei2_upper manhattan', 'nei2_east bronx', 'nei2_other', 'nei2_midtown manhattan',
             'nei2_southeastern queens', 'nei2_southwestern brooklyn', 'nei2_southwestern queens', 'nei2_rockaway peninsula', 'nei2_northern brooklyn',
             'nei2_downtown manhattan', 'nei2_southern brooklyn', 'nei3_not_mapped_yet', 'nei3_manhattan', 'nei3_brooklyn', 'nei3_bronx', 'nei3_queens']
"""
