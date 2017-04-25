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
INDEX_COPY = 'index_copy'

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
rent_file = '../../data/neis_from_renthop_lower.json'

seeds_fp = '../../seeds.json'
splits_big_fp='../../splits_big.json'
splits_small_fp='../../splits_small.json'

magic_file = '../../data/redhoop/listing_image_time.csv'

# train_file = '../data/redhoop/train.json'
# test_file = '../data/redhoop/test.json'
# train_geo_file = '../data/redhoop/with_geo/train_geo.json'
# test_geo_file = '../data/redhoop/with_geo/test_geo.json'
# rent_file = 'with_geo/data/neis_from_renthop_lower.json'
# seeds_fp = '../../seeds.json'
# splits_big_fp='../../splits_big.json'
# splits_small_fp='../../splits_small.json'


#########################################################################################
# Mongo Control
#########################################################################################

SEEDS = json.load(open(seeds_fp))
SPLITS_BIG=json.load(open(splits_big_fp))
SPLITS_SMALL=json.load(open(splits_small_fp))


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
    N=N%5
    return df.loc[SPLITS_BIG[N],:], df.loc[SPLITS_SMALL[N], :]


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
    df[INDEX_COPY] = df.index.values

    return df

def fix_index(df):
    df.index = df[INDEX_COPY]


#########################################################################################
# loading data
#########################################################################################

#########################################################################################
# Creating Neis
#########################################################################################
def normalize_bed_bath(df):
    df['bed_bath']=df[[BEDROOMS, BATHROOMS]].apply(lambda s: (s[BEDROOMS], s[BATHROOMS]), axis=1)
    def norm(s):
        bed=s[0]
        bath=s[1]
        if bed==0:
            if bath>=1.5:
                return [0,2.0]
        elif bed==1:
            if bath>=2.5:
                return [1,2.0]
        elif bed==2:
            if bath>=3.0:
                return [2,3.0]
        elif bed==3:
            if bath>=4.0:
                return [3,4.0]
        elif bed==4:
            if bath==0:
                return [4,1]
            elif bath>=4.5:
                return [4,4.5]
        elif bed>=5:
            if bath <=1.5:
                return [5,1.5]
            elif bath <=2.5:
                return [5,2.5]
            elif bath <=3.5:
                return [5,3]
            else:
                return [5,4]

        return [bed, bath]

    df['bed_bath']=df['bed_bath'].apply(norm)
    df[BED_NORMALIZED]=df['bed_bath'].apply(lambda s:s[0])
    df[BATH_NORMALIZED]=df['bed_bath'].apply(lambda s:s[1])


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


#########################################################################################
# MNGR HCC
#########################################################################################

def hcc_encode(train_df, test_df, variable, binary_target, k=5, f=1, g=1, r_k=0.01, folds=5):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    prior_prob = train_df[binary_target].mean()
    hcc_name = "_".join(["hcc", variable, binary_target])

    seed = int(time())
    print 'seed hcc {}'.format(seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
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


def process_mngr_categ_preprocessing(train_df, test_df):
    col = MANAGER_ID
    new_cols = []
    for df in [train_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
        df['target_low'] = df[TARGET].apply(lambda s: 1 if s == 'low' else 0)
    for binary_col in ['target_high', 'target_medium', 'target_low']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols
#########################################################################################
# MNGR HCC
#########################################################################################

#########################################################################################
# MNGR NUM
#########################################################################################
def process_manager_num(train_df, test_df):
    mngr_num_col = 'manager_num'
    df = train_df.groupby(MANAGER_ID)[MANAGER_ID].count()
    # df[df<=1]=-1
    df = df.apply(float)
    df = df.to_frame(mngr_num_col)
    train_df = pd.merge(train_df, df, left_on=MANAGER_ID, right_index=True)
    test_df = pd.merge(test_df, df, left_on=MANAGER_ID, right_index=True, how='left')

    return train_df, test_df, [mngr_num_col]

#########################################################################################
# MNGR NUM
#########################################################################################


#########################################################################################
# BID HCC
#########################################################################################

def process_bid_categ_preprocessing(train_df, test_df):
    col = BUILDING_ID
    new_cols = []
    for df in [train_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
        df['target_low'] = df[TARGET].apply(lambda s: 1 if s == 'low' else 0)
    for binary_col in ['target_high', 'target_medium', 'target_low']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols
#########################################################################################
# BID HCC
#########################################################################################


#########################################################################################
# BID NUM
#########################################################################################

def process_bid_num(train_df, test_df):
    bid_num_col = 'bid_num'
    df = train_df.groupby(BUILDING_ID)[BUILDING_ID].count()
    # df[df<=1]=-1
    df = df.apply(float)
    df = df.to_frame(bid_num_col)
    train_df = pd.merge(train_df, df, left_on=BUILDING_ID, right_index=True)
    test_df = pd.merge(test_df, df, left_on=BUILDING_ID, right_index=True, how='left')

    return train_df, test_df, [bid_num_col]

#########################################################################################
# BID NUM
#########################################################################################

#########################################################################################
# Listing id
#########################################################################################

def process_listing_id(train_df, test_df):
    return train_df, test_df, [LISTING_ID]
#########################################################################################
# Listing id
#########################################################################################


#########################################################################################
# NEI 123
#########################################################################################

def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)


def get_dummy_cols(col_name, col_values):
    return ['{}_{}'.format(col_name, val) for val in col_values]

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

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols
#########################################################################################
# NEI 123
#########################################################################################


#########################################################################################
# MNGR AVG PRICE
#########################################################################################

def process_mngr_avg_median_price(train_df, test_df):
    df = pd.concat([train_df, test_df])
    bed_bath_median = 'bed_bath_median'
    df[bed_bath_median] = df.groupby([BED_NORMALIZED, BATH_NORMALIZED])[PRICE].transform('median')

    bed_bath_diff = 'bed_bath_diff'
    df[bed_bath_diff]=df[PRICE]-df[bed_bath_median]

    bed_bath_raio = 'bed_bath_ratio'
    df[bed_bath_raio]=df[bed_bath_diff]/df['bed_bath_median']

    group_by = df.groupby(MANAGER_ID)[bed_bath_diff]
    df['gr_by_mngr_bed_bath_diff_median']= group_by.transform('median')
    df['gr_by_mngr_bed_bath_diff_quantile_0.25']= group_by.transform('quantile', 0.25)
    df['gr_by_mngr_bed_bath_diff_quantile_0.75']= group_by.transform('quantile', 0.75)
    df['gr_by_mngr_bed_bath_diff_mean']= group_by.transform('mean')

    group_by = df.groupby(MANAGER_ID)[bed_bath_raio]
    df['gr_by_mngr_bed_bath_ratio_median']= group_by.transform('median')
    df['gr_by_mngr_bed_bath_ratio_quantile_0.25']= group_by.transform('quantile', 0.25)
    df['gr_by_mngr_bed_bath_ratio_quantile_0.75']= group_by.transform('quantile', 0.75)
    df['gr_by_mngr_bed_bath_ratio_mean']= group_by.transform('mean')

    new_cols= ['bed_bath_diff','bed_bath_ratio','bed_bath_median',
               'gr_by_mngr_bed_bath_diff_median','gr_by_mngr_bed_bath_diff_mean',
               'gr_by_mngr_bed_bath_diff_quantile_0.25','gr_by_mngr_bed_bath_diff_quantile_0.75',
               'gr_by_mngr_bed_bath_ratio_median', 'gr_by_mngr_bed_bath_ratio_mean' ,
               'gr_by_mngr_bed_bath_ratio_quantile_0.25', 'gr_by_mngr_bed_bath_ratio_quantile_0.75'
               ]

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols

#########################################################################################
# MNGR AVG PRICE
#########################################################################################




#########################################################################################
# OTHER MEDIANS
#########################################################################################
def process_other_mngr_medians(train_df, test_df):
    features = ['num_features', 'num_photos', 'word_num_in_descr', BED_NORMALIZED, BATH_NORMALIZED]
    df = pd.concat([train_df, test_df])
    new_cols = []
    for f in features:
        col = 'get_by_mngr_{}_mean'.format(f)
        df[col] = df.groupby(MANAGER_ID)[f].transform('mean')
        new_cols.append(col)
        if f in [BATH_NORMALIZED, BED_NORMALIZED]:
            continue

        col = 'get_by_mngr_{}_median'.format(f)
        new_cols.append(col)
        df[col] = df.groupby(MANAGER_ID)[f].transform('median')

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols


#########################################################################################
# OTHER MEDIANS
#########################################################################################


#########################################################################################
# OTHER MEDIANS nEW
#########################################################################################

def get_main_value(s):
    n = int(0.66*len(s))
    vals = {k:0 for k in set(s)}
    for x in s:
        vals[x]+=1

    for k,v in vals.iteritems():
        if v>=n:
            return k

def process_other_mngr_medians_new(train_df, test_df):
    df = pd.concat([train_df, test_df])
    total_minutes_col='total_minutes'
    df[total_minutes_col] = df[CREATED_MINUTE]+24*df[CREATED_HOUR]
    features = [PRICE, LATITUDE, LONGITUDE, total_minutes_col]
    new_cols = []
    for f in features:
        col = 'get_by_mngr_{}_mean'.format(f)
        df[col] = df.groupby(MANAGER_ID)[f].transform('mean')
        new_cols.append(col)

        col = 'get_by_mngr_{}_median'.format(f)
        new_cols.append(col)
        df[col] = df.groupby(MANAGER_ID)[f].transform('median')

    main_hour='main_hour'
    bl = df.groupby(MANAGER_ID)[CREATED_HOUR].apply(get_main_value).to_frame(main_hour)
    df = pd.merge(df, bl, left_on=MANAGER_ID, right_index=True)
    new_cols.append(main_hour)

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)
    return train_df, test_df, new_cols
#########################################################################################
# OTHER MEDIANS NEW
#########################################################################################


#########################################################################################
#  FEATURES
#########################################################################################
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

#########################################################################################
#  FEATURES
#########################################################################################

####################################################
#MAGIC
#######################################################

def process_magic(train_df, test_df):
    image_date = pd.read_csv(magic_file)
    image_date.loc[80240,"time_stamp"] = 1478129766
    # image_date.loc[image_date['Listing_Id']==7119094, "time_stamp"] = 1478129766
    image_date["img_date"] = pd.to_datetime(image_date["time_stamp"], unit="s")
    image_date["img_days_passed"] = (image_date["img_date"].max() - image_date["img_date"]).astype(
        "timedelta64[D]").astype(int)
    image_date["img_date_month"] = image_date["img_date"].dt.month
    image_date["img_date_week"] = image_date["img_date"].dt.week
    image_date["img_date_day"] = image_date["img_date"].dt.day
    image_date["img_date_dayofweek"] = image_date["img_date"].dt.dayofweek
    image_date["img_date_dayofyear"] = image_date["img_date"].dt.dayofyear
    image_date["img_date_hour"] = image_date["img_date"].dt.hour
    image_date["img_date_minute"] = image_date["img_date"].dt.minute
    image_date["img_date_second"] = image_date["img_date"].dt.second
    image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(
        lambda x: 1 if x < 10 else 2 if x < 20 else 3)

    df = pd.concat([train_df, test_df])
    df = pd.merge(df, image_date, left_on=LISTING_ID, right_on='Listing_Id')
    new_cols = ["img_days_passed","img_date_month","img_date_week",
                "img_date_day","img_date_dayofweek","img_date_dayofyear",
                "img_date_hour", "img_date_monthBeginMidEnd",
                "img_date_minute", "img_date_second"]#+["img_date", "time_stamp"]

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols

####################################################
#MAGIC
#######################################################

####################################################
#FEATURES NEW
#######################################################
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd


def process_features_new(X_train, X_test):
    fmt = lambda feat: [s.replace("\u00a0", "").strip().lower().replace(" ", "_") for s in feat]  # format features
    X_train["features"] = X_train["features"].apply(fmt)
    X_test["features"] = X_test["features"].apply(fmt)
    features = [f for f_list in list(X_train["features"]) + list(X_test["features"]) for f in f_list]
    ps = pd.Series(features)
    grouped = ps.groupby(ps).agg(len)
    features = grouped[grouped >= 10].index.sort_values().values    # limit to features with >=10 observations
    mlb = MultiLabelBinarizer().fit([features])
    columns = ['feature_' + s for s in mlb.classes_]
    flt = lambda l: [i for i in l if i in mlb.classes_]     # filter out features not present in MultiLabelBinarizer
    X_train = X_train.join(pd.DataFrame(data=mlb.transform(X_train["features"].apply(flt)), columns=columns, index=X_train.index))
    X_test = X_test.join(pd.DataFrame(data=mlb.transform(X_test["features"].apply(flt)), columns=columns, index=X_test.index))

    return X_train, X_test, columns

####################################################
#FEATURES NEW
#######################################################

####################################################
#BID AVG
#######################################################

BED_BATH_DIFF = 'bed_bath_diff'
BED_BATH_RATIO = 'bed_bath_ratio'


def process_bid_prices_medians(train_df, test_df):
    df = pd.concat([train_df, test_df])
    bid__price_ratio_median = 'bid__price_ratio_median'
    bid__price_ratio_mean = 'bid__price_ratio_mean'

    bid__price_diff_median = 'bid__price_diff_median'
    bid__price_diff_mean = 'bid__price_diff_mean'

    group_by = df.groupby(BUILDING_ID)

    df[bid__price_ratio_median] = group_by[BED_BATH_RATIO].transform('median')
    df[bid__price_ratio_mean] = group_by[BED_BATH_RATIO].transform('mean')

    df[bid__price_diff_median] = group_by[BED_BATH_DIFF].transform('median')
    df[bid__price_diff_mean] = group_by[BED_BATH_DIFF].transform('mean')

    bid_bias_price_diff = 'bid_bias_price_diff'
    bid_bias_price_ratio = 'bid_bias_price_ratio'

    df[bid_bias_price_diff] = df[BED_BATH_DIFF] - df[bid__price_diff_median]
    df[bid_bias_price_ratio] = df[BED_BATH_RATIO] / df[bid__price_ratio_median]

    new_cols = [
        bid__price_ratio_median,
        bid__price_ratio_mean,
        bid__price_diff_median,
        bid__price_diff_mean,
        bid_bias_price_diff,
        bid_bias_price_ratio
    ]

    features_to_avg = ['num_features', 'num_photos', 'word_num_in_descr']
    for f in features_to_avg:
        col = 'get_by_bid_{}_median'.format(f)
        new_cols.append(col)
        df[col] = group_by[f].transform('median')

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols

####################################################
#BID AVG
#######################################################

####################################################
#STREET AVG
#######################################################
BED_BATH_DIFF = 'bed_bath_diff'
BED_BATH_RATIO = 'bed_bath_ratio'
DISPLAY_ADDRESS = 'display_address'
NORMALIZED_DISPLAY_ADDRESS = 'normalized_display_address'
MANAGER_ID = 'manager_id'

def reverse_norm_map(m):
    res = {}
    for k, v in m.iteritems():
        for s in v:
            res[s.lower()] = k.lower()

    return res


NORMALIZATION_MAP = {
    'street': ['St', 'St.', 'Street', 'St,', 'st..', 'street.'],
    'avenue': ['Avenue', 'Ave', 'Ave.'],
    'square': ['Square'],
    'east': ['e', 'east', 'e.'],
    'west': ['w', 'west', 'w.'],
    'road':['road', 'rd', 'rd.']
}

REVERSE_NORM_MAP = reverse_norm_map(NORMALIZATION_MAP)


# Fifth, Third

def normalize_tokens(s):
    tokens = s.split()
    for i in range(len(tokens)):
        tokens[i] = if_starts_with_digit_return_digit_prefix(tokens[i])
        t = tokens[i]
        if t.lower() in REVERSE_NORM_MAP:
            tokens[i] = REVERSE_NORM_MAP[t.lower()]
    return ' '.join(tokens)

def if_starts_with_digit_return_digit_prefix(s):
    if not s[0].isdigit():
        return s
    last=0
    for i in range(len(s)):
        if s[i].isdigit():
            last=i
        else:
            break

    return s[0:last+1]


def normalize_string(s):
    s = normalize_tokens(s)
    if s == '':
        return s

    s=s.lower()

    tokens = s.split()
    if len(tokens) == 2:
        return ' '.join(tokens)
    if tokens[0].replace('.', '').replace('-', '').isdigit():
        return ' '.join(tokens[1:])
    else:
        return ' '.join(tokens)

def normalize_display_address_df(df):
    df[NORMALIZED_DISPLAY_ADDRESS] = df[DISPLAY_ADDRESS].apply(normalize_string)

def process_street_counts(train_df, test_df):
    df = pd.concat([train_df, test_df])
    normalize_display_address_df(df)
    col = 'street_popularity'
    df[col] = df.groupby(NORMALIZED_DISPLAY_ADDRESS)[MANAGER_ID].transform('count')

    new_cols=[col]
    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols


def process_street_prices_medians(train_df, test_df):
    df = pd.concat([train_df, test_df])
    normalize_display_address_df(df)
    street__price_ratio_median = 'street__price_ratio_median'
    street__price_ratio_mean = 'street__price_ratio_mean'

    street__price_diff_median = 'street__price_diff_median'
    street__price_diff_mean = 'street__price_diff_mean'

    group_by = df.groupby(NORMALIZED_DISPLAY_ADDRESS)

    df[street__price_ratio_median] = group_by[BED_BATH_RATIO].transform('median')
    df[street__price_ratio_mean] = group_by[BED_BATH_RATIO].transform('mean')

    df[street__price_diff_median] = group_by[BED_BATH_DIFF].transform('median')
    df[street__price_diff_mean] = group_by[BED_BATH_DIFF].transform('mean')

    street_bias_price_diff = 'street_bias_price_diff'
    street_bias_price_ratio = 'street_bias_price_ratio'

    df[street_bias_price_diff] = df[BED_BATH_DIFF] - df[street__price_diff_median]
    df[street_bias_price_ratio] = df[BED_BATH_RATIO] / df[street__price_ratio_median]

    new_cols = [
        street__price_ratio_median,
        street__price_ratio_mean,
        street__price_diff_median,
        street__price_diff_mean,
        street_bias_price_diff,
        street_bias_price_ratio
    ]

    features_to_avg = ['num_features', 'num_photos', 'word_num_in_descr']
    for f in features_to_avg:
        col = 'get_by_street_{}_median'.format(f)
        new_cols.append(col)
        df[col] = group_by[f].transform('median')

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols

####################################################
#STREET AVG
#######################################################

#########################################################################################
#  NAIVE STATS
#########################################################################################

def process_mngr_target_ratios(train_df, test_df):
    return process_target_ratios(train_df, test_df, MANAGER_ID, 5)

def process_bid_target_ratios(train_df, test_df):
    return process_target_ratios(train_df, test_df, BUILDING_ID, 5)



def process_target_ratios(train_df, test_df, col, folds):
    seed = int(time())
    print 'seed naive_stats {}'.format(seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    target_vals = ['high', 'medium', 'low']
    new_cols = {k: '{}_target_ratios_{}'.format(col, k) for k in target_vals}
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        calc_target_ratios(big, small, col, new_cols, train_df)

    calc_target_ratios(train_df.copy(), test_df.copy(),col, new_cols, update_df=test_df)

    return train_df, test_df, new_cols.values()


def calc_target_ratios(big, small, col, new_cols, update_df):
    target_vals = ['high', 'medium', 'low']
    dummies = {k:'target_cp_{}'.format(k) for k in target_vals}

    big['target_cp'] = big[TARGET].copy()
    big= pd.get_dummies(big, columns=['target_cp'])
    grouped = big.groupby(col).mean()
    small = pd.merge(small, grouped[dummies.values()], left_on=col, right_index=True)
    for t in target_vals:
        new_col = new_cols[t]
        update_df.loc[small.index, new_col] = small[dummies[t]]

#########################################################################################
#  NAIVE STATS
#########################################################################################

####################################################
#weighted price ratio
#######################################################
BED_BATH_MEDIAN= 'bed_bath_median'
BED_BATH_DIFF = 'bed_bath_diff'
BED_BATH_RATIO = 'bed_bath_ratio'

def process_mngr_weighted_price_ratio(train_df, test_df):
    return process_weighted_price_ratio(train_df, test_df, MANAGER_ID, 5)



def process_weighted_price_ratio(train_df, test_df, col, folds):
    skf = StratifiedKFold(folds)
    target_vals = ['high', 'medium', 'low']
    new_cols = ['weighted_price_ratio', 'weighted_price_diff']
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        calc_weighted_price_ratio(big, small, col, train_df)

    calc_weighted_price_ratio(train_df.copy(), test_df.copy(), col, update_df=test_df)

    return train_df, test_df, new_cols


def calc_weighted_price_ratio(big, small, col, update_df):
    target_vals = ['high', 'medium', 'low']
    new_cols = ['weighted_price_ratio', 'weighted_price_diff']

    big['target_cp'] = big[TARGET].copy()
    big= pd.get_dummies(big, columns=['target_cp'])
    big['weighted_price_ratio'] = 3*big['target_cp_high']*big[BED_BATH_RATIO]+big['target_cp_medium']*big[BED_BATH_RATIO]
    big['weighted_price_diff'] = 3*big['target_cp_high']*big[BED_BATH_DIFF]+big['target_cp_medium']*big[BED_BATH_DIFF]

    grouped = big.groupby(col).mean()
    small = pd.merge(small[[col]], grouped[new_cols], left_on=col, right_index=True)
    for new_col in new_cols:
        update_df.loc[small.index, new_col] = small[new_col]


####################################################
#weighted price ratio
####################################################


####################################################
#DISTANCE TO CENTER
####################################################

from haversine import haversine

def process_distance_to_center(train_df, test_df):
    df = pd.concat([train_df, test_df])
    col='distance_to_center'
    lat=df[LATITUDE].median()
    long = df[LONGITUDE].median()

    df[col] = df[[LATITUDE, LONGITUDE]].apply(lambda s: haversine((lat, long), (s[0], s[1])), axis=1)

    new_cols=[col]
    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols
####################################################
#DISTANCE TO CENTER
####################################################


def shuffle_df(df):
    return df.iloc[np.random.permutation(len(df))]


def get_loss_at1K(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    return results_on_test[999]


def loss_with_per_tree_stats(train_df, test_df, new_cols):
    features, test_df, train_df = process_split(train_df, test_df, new_cols)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values

    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values
    print features

    seed = int(time())
    print 'XGB seed {}'.format(seed)
    estimator = xgb.XGBClassifier(n_estimators=1000,
                                  objective='mlogloss',
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  seed=seed)
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

    train_df, test_df, new_cols = process_mngr_target_ratios(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_bid_target_ratios(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols


    return features, test_df, train_df


def process_all_name(train_df, test_df):
    features = ['bathrooms', 'bedrooms', 'latitude',
                'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_month", "created_day",
                CREATED_HOUR, CREATED_MINUTE, DAY_OF_WEEK]

    train_df, test_df, new_cols = process_manager_num(train_df, test_df)
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

    train_df, test_df, new_cols = process_mngr_avg_median_price(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols


    train_df, test_df, new_cols = process_other_mngr_medians(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols


    train_df, test_df, new_cols = process_other_mngr_medians_new(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_magic(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_bid_prices_medians(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features+=new_cols

    train_df, test_df, new_cols = process_street_counts(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_street_prices_medians(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_mngr_weighted_price_ratio(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_distance_to_center(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features += new_cols

    train_df, test_df, new_cols = process_features_new(train_df, test_df)
    train_df, test_df = shuffle_df(train_df), shuffle_df(test_df)
    features+=new_cols

    return train_df, test_df, features


def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    results_on_train = estimator.evals_result()['validation_0']['mlogloss']
    return {
        'train': results_on_train,
        'test': results_on_test
    }


def do_test_xgboost(name, mongo_host, experiment_max_time=15*60):
    all_losses = []
    l_results_per_tree = []
    losses_at_1K = []

    train_df = load_train()
    test_df = load_test()

    train_df, test_df, features = process_all_name(train_df, test_df)
    fix_index(train_df)
    fix_index(test_df)

    ii_importance = []
    for counter in range(5):
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



do_test_xgboost('stacking_new_heu_all', '35.187.46.132')
