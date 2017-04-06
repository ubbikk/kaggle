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

TARGET = 'interest_level'
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
F_COL = 'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE = 'created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
CREATED='created'
LABEL='lbl'




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







