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
