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


# def process_other_mngr_medians(train_df, test_df):
#     features = ['num_features', 'num_photos', 'word_num_in_descr', BED_NORMALIZED, BATH_NORMALIZED]
#     df = pd.concat([train_df, test_df])
#     new_cols = []
#     for f in features:
#         col = 'get_by_mngr_{}_mean'.format(f)
#         df[col] = df.groupby(MANAGER_ID)[f].transform('mean')
#         new_cols.append(col)
#         if f in [BATH_NORMALIZED, BED_NORMALIZED]:
#             continue
#
#         col = 'get_by_mngr_{}_median'.format(f)
#         new_cols.append(col)
#         df[col] = df.groupby(MANAGER_ID)[f].transform('median')
#
#     df_to_merge = df[[LISTING_ID] + new_cols]
#     train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
#     test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)
#
#     return train_df, test_df, new_cols


    # def get_main_value(s):
    #     n = int(0.66*len(s))
    #     vals = {k:0 for k in set(s)}
    #     for x in s:
    #         vals[x]+=1
    #
    #     for k,v in vals.iteritems():
    #         if v>=n:
    #             return k
    #
    # def process_other_mngr_medians_new(train_df, test_df):
    #     df = pd.concat([train_df, test_df])
    #     total_minutes_col='total_minutes'
    #     df[total_minutes_col] = df[CREATED_MINUTE]+24*df[CREATED_HOUR]
    #     features = [PRICE, LATITUDE, LONGITUDE, total_minutes_col]
    #     new_cols = []
    #     for f in features:
    #         col = 'get_by_mngr_{}_mean'.format(f)
    #         df[col] = df.groupby(MANAGER_ID)[f].transform('mean')
    #         new_cols.append(col)
    #
    #         col = 'get_by_mngr_{}_median'.format(f)
    #         new_cols.append(col)
    #         df[col] = df.groupby(MANAGER_ID)[f].transform('median')
    #
    #     main_hour='main_hour'
    #     bl = df.groupby(MANAGER_ID)[CREATED_HOUR].apply(get_main_value).to_frame(main_hour)
    #     df = pd.merge(df, bl, left_on=MANAGER_ID, right_index=True)
    #     new_cols.append(main_hour)
    #
    #     df_to_merge = df[[LISTING_ID] + new_cols]
    #     train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    #     test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)
    #     return train_df, test_df, new_cols
