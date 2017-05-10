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
LABEL = 'ltrain_df'
BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'
COL = 'normalized_features'
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'
NEI = 'neighbourhood'
BORO = 'boro'
INDEX_COPY = 'index_copy'

def process_dummy_magic(train_df, test_df):
    col="img_date_dayofyear"
    df = pd.concat([train_df, test_df])
    df['magic']=df[col]
    magics = set(df['magic'])
    df = pd.get_dummies(df, columns=['magic'])
    cols = ['magic_{}'.format(x) for x in magics]
    groupby = df.groupby(MANAGER_ID)
    new_cols = []
    for col in cols:
        df['mngr_{}'.format(col)]=groupby[col].transform('mean')
        new_cols.append('mngr_{}'.format(col))

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols

def process_agregated_mngr_magic(train_df, test_df):
    col="img_date_dayofyear"
    train_df['t'] = train_df[TARGET]
    train_df = pd.get_dummies(train_df, columns=['t'])
    train_df['l'] = train_df.groupby(col)['t_low'].transform('mean')
    train_df['m'] = train_df.groupby(col)['t_medium'].transform('mean')
    train_df['h'] = train_df.groupby(col)['t_high'].transform('mean')

    blja = train_df.groupby(MANAGER_ID)[['l', 'm', 'h']].transform('mean')
    for c in ['l', 'm', 'h']:
        del train_df[c]

    train_df=pd.merge(train_df, blja, how='left', left_on=MANAGER_ID, right_index=True)
    test_df=pd.merge(test_df, blja, how='left', left_on=MANAGER_ID, right_index=True)

    return train_df, test_df, ['l', 'm', 'h']
