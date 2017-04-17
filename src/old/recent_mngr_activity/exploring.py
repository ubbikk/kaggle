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
from sklearn.cross_validation import cross_val_score, KFold
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.spatial import KDTree
import math



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
CREATED='created'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
NEI = 'neighbourhood'
BORO = 'boro'
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'

rent_file = 'with_geo/data/neis_from_renthop_lower.json'

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
    s=s.lower()
    rent = load_rent()
    if s in rent:
        return rent[s]

    if s in EXACT_MAP:
        return rent[EXACT_MAP[s]]

    if s in SPECIAL:
        return SPECIAL[s]

    return ('not_mapped_yet', 'not_mapped_yet', 'not_mapped_yet')



FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)

train_file = '../data/redhoop/train.json'
test_file = '../data/redhoop/test.json'

train_geo_file = '../data/redhoop/with_geo/train_geo.json'
test_geo_file = '../data/redhoop/with_geo/test_geo.json'


# train_file = '../../data/redhoop/train.json'
# test_file = '../../data/redhoop/test.json'
#
# train_geo_file = '../../data/redhoop/with_geo/train_geo.json'
# test_geo_file = '../../data/redhoop/with_geo/test_geo.json'

def load_df(file, geo_file):
    df = pd.read_json(file)
    geo = pd.read_json(geo_file)
    df[NEI]= geo[NEI]
    df['tmp']=df[NEI].apply(transform_geo_to_rent)
    df[NEI_1]=df['tmp'].apply(lambda s:None if s is None else s[0])
    df[NEI_2]=df['tmp'].apply(lambda s:None if s is None else s[1])
    df[NEI_3]=df['tmp'].apply(lambda s:None if s is None else s[2])
    return basic_preprocess(df)


def load_train():
    return load_df(train_file, train_geo_file)


def load_test():
    return load_df(test_file, test_geo_file)


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




BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'

def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)

def get_dummy_cols(col_name, col_values):
    return ['{}_{}'.format(col_name, val) for val in col_values]


def normalize_bed_bath(df):
    df[BED_NORMALIZED] = df[BEDROOMS].apply(lambda s: s if s<=3 else 3)
    def norm_bath(s):
        s=round(s)
        if s==0:
            return 1
        if s>=2:
            return 2
        return s

    df[BATH_NORMALIZED]=df[BATHROOMS].apply(norm_bath)




def process_nei123(train_df, test_df):
    df = pd.concat([train_df, test_df])
    normalize_bed_bath(df)
    sz= float(len(df))
    # neis_cols = [NEI_1, NEI_2, NEI_3]
    new_cols=[]
    for col in [NEI_1, NEI_2]:
        new_col = 'freq_of_{}'.format(col)
        df[new_col] = df.groupby(col)[PRICE].transform('count')
        df[new_col] = df[new_col]/sz
        new_cols.append(new_col)

    beds_vals =[0,1,2,3]
    for col in [NEI_1, NEI_2, NEI_3]:
        for bed in beds_vals:
            new_col = 'freq_of_{}, with bed={}'.format(col, bed)
            df[new_col] = df.groupby([col, BED_NORMALIZED])[PRICE].transform('count')
            df[new_col] = df[new_col]/sz
            new_cols.append(new_col)

    for col in [NEI_1, NEI_2]:
        new_col = 'median_ratio_of_{}'.format(col)
        df['tmp'] = df.groupby([col, BEDROOMS])[PRICE].transform('median')
        df[new_col] = df[PRICE]-df['tmp']
        df[new_col] = df[new_col]/df['tmp']
        new_cols.append(new_col)
    for col in [NEI_1, NEI_2, NEI_3]:
        vals = set(df[col])
        if None in vals:
            vals.remove(None)
        df = pd.get_dummies(df, columns=[col])
        dummies= get_dummy_cols(col, vals)
        new_cols+=dummies

    for d in [train_df, test_df]:
        for col in new_cols:
            d[col]=df.loc[d.index, col]

    return train_df, test_df, new_cols



train_df, test_df = load_train(), load_test()
train_df, test_df, new_cols = process_nei123(train_df, test_df)

