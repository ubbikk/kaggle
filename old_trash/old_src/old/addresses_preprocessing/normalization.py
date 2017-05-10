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

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)

train_file = '../data/redhoop/train.json'
test_file = '../data/redhoop/test.json'


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


def normalize(s):
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


def test_blja():
    train_df = load_train()
    bl = train_df[DISPLAY_ADDRESS].to_frame()
    bl['norm'] = bl[DISPLAY_ADDRESS].apply(normalize)


def load_train():
    return basic_preprocess(pd.read_json(train_file))


def load_test():
    return basic_preprocess(pd.read_json(test_file))


def test():
    train_df = load_train()
    bl = train_df[DISPLAY_ADDRESS].to_dict().values()
    count_map = {}
    for txt in bl:
        for token in txt.split():
            if token in count_map:
                count_map[token] += 1
            else:
                count_map[token] = 1

    bl = [(k, v) for k, v in count_map.iteritems()]
    bl.sort(key=lambda s: s[1], reverse=True)
    return count_map


# test_blja()
