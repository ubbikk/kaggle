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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

def add_log_reg_cols(train_df, test_df, variable, folds, beans):
    skf = StratifiedKFold(folds)
    prior = pd.get_dummies(train_df, columns=[TARGET])[['interest_level_high', 'interest_level_medium', 'interest_level_low']].mean()
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df[TARGET]):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        add_log_reg_cols_big_small(big, small, prior, train_df, variable, beans)

    add_log_reg_cols_big_small(train_df, test_df, prior, test_df, variable, beans)



def add_log_reg_cols_big_small(big, small, prior, update_df, variable, beans):
    df = pd.concat([big, small])
    df['mngr_cnt'] = df.groupby(variable)[variable].transform('count')
    df, beans_cols = bean_df(df, 'mngr_cnt', beans)
    big[beans_cols]=df.loc[big.index, beans_cols]
    small[beans_cols]=df.loc[small.index, beans_cols]

    big['t'] = big[TARGET]
    big = pd.get_dummies(big, columns=['t'])
    agg = OrderedDict([
        (variable, {'count': 'count'}),
        ('t_high', {'high': 'mean'}),
        ('t_medium', {'medium': 'mean'}),
        ('t_low', {'low': 'mean'})
    ])
    df = big.groupby(variable).agg(agg)
    cols = ['man_id_high', 'man_id_medium', 'man_id_low']
    df.columns = cols
    big = pd.merge(big, df, left_on=variable, right_index=True)
    small = pd.merge(small, df,left_on=variable, right_index=True, how='left')
    small.loc[small['man_id_count'].isnull(), cols] = [0] + [x for x in prior]
    big_arr = big[['man_id_high', 'man_id_medium', 'man_id_low']+beans_cols]
    small_arr = small[['man_id_high', 'man_id_medium', 'man_id_low']+beans_cols]
    target_vals = ['high', 'medium', 'low']
    for t in target_vals:
        big_target = big[TARGET].apply(lambda s: 1 if s == t else 0)
        small_target = small[TARGET].apply(lambda s: 1 if s == t else 0)
        model = LogisticRegression()
        model.fit(big_arr, big_target)
        proba = model.predict_proba(small_arr)[:, 1]
        auc = roc_auc_score(small_target, proba)
        print 'auc={}'.format(auc)
        update_df.loc[small.index, 'log_reg_{}'.format(t)] = proba


def bean_df(df, col, beans):
    def transform(s):
        if s<beans[0]:
            return '(, {})'.format(beans[0])
        for j in range(len(beans)-1):
            if s>=beans[j] and s<beans[j+1]:
                if beans[j+1]-beans[j]==1:
                    return str(beans[j])
                else:
                    return '[{}, {})'.format(beans[j], beans[j+1])
        return '[{}, )'.format(beans[len(beans)-1])

    tmp = '{}_bean'.format(col)
    df[tmp]= df[col].apply(transform)
    new_cols = set(df[tmp])
    df = pd.get_dummies(df, columns=[tmp])
    new_cols=['{}_{}'.format(tmp, x) for x in new_cols]
    return df, new_cols


def process_mngr_ens_and_beans(train_df, test_df):
    col = MANAGER_ID
    folds = 5
    beans = range(1, 25)+range(25, 100, 5)+range(100, 200, 10)+range(200, 1100, 100)
    add_log_reg_cols(train_df, test_df, col, folds, beans)
    return train_df, test_df, ['log_reg_{}'.format(t) for t in ['high', 'medium', 'low']]
