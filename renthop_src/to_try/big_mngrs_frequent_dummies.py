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



def process_frequent_dummies(df, col, num):
    df['num'] = df.groupby(col)[PRICE].transform('count')
    small = df[df['num'] >= num][[col]]
    bl = pd.get_dummies(small, columns=[col])

    df = pd.merge(df, bl, left_index=True, right_index=True, how='left')
    new_columns = list(bl.columns.values)
    df.loc[df[df[new_columns[0]].isnull()].index,new_columns]=0
    return df, new_columns

def process_mngr_freaquent_dummies(train_df, test_df):
    df = pd.concat([train_df, test_df])
    df, new_cols = process_frequent_dummies(df, MANAGER_ID, 100)

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols

def process_bid_freaquent_dummies(train_df, test_df):
    df = pd.concat([train_df, test_df])
    df, new_cols = process_frequent_dummies(df, BUILDING_ID, 100)

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols