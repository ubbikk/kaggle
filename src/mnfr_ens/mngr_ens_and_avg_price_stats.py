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
F_COL = u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE = 'created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
NEI = 'neighbourhood'
BORO = 'boro'
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'
gr_by_mngr_bed_bath_diff_median = 'gr_by_mngr_bed_bath_diff_median'
gr_by_mngr_bed_bath_diff_mean = 'gr_by_mngr_bed_bath_diff_mean'
gr_by_mngr_bed_bath_ratio_median = 'gr_by_mngr_bed_bath_ratio_median'
gr_by_mngr_bed_bath_ratio_mean = 'gr_by_mngr_bed_bath_ratio_mean'
bed_bath_diff = 'bed_bath_diff'
bed_bath_ratio = 'bed_bath_ratio'


def add_log_reg_cols(train_df, test_df, variable, folds):
    skf = StratifiedKFold(folds)
    prior = pd.get_dummies(train_df, columns=[TARGET])[
        ['interest_level_high', 'interest_level_medium', 'interest_level_low']].mean()
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df[TARGET]):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        add_log_reg_cols_big_small(big, small, prior, train_df, variable)

    add_log_reg_cols_big_small(train_df, test_df, prior, test_df, variable)


def add_log_reg_cols_big_small(big, small, prior, update_df, variable):
    big['t'] = big[TARGET]
    big = pd.get_dummies(big, columns=['t'])
    agg = OrderedDict([
        (variable, {'count': 'count'}),
        ('t_high', {'high': 'mean'}),
        ('t_medium', {'medium': 'mean'}),
        ('t_low', {'low': 'mean'})
    ])
    df = big.groupby(variable).agg(agg)
    cols = ['man_id_count', 'man_id_high', 'man_id_medium', 'man_id_low']
    df.columns = cols
    big = pd.merge(big, df, left_on=variable, right_index=True)
    small = pd.merge(small, df, left_on=variable, right_index=True, how='left')
    small.loc[small['man_id_count'].isnull(), cols] = [0] + [x for x in prior]

    cols_to_use = ['man_id_high', 'man_id_medium', 'man_id_low',
                   gr_by_mngr_bed_bath_diff_median, gr_by_mngr_bed_bath_diff_mean,
                   gr_by_mngr_bed_bath_ratio_median, gr_by_mngr_bed_bath_ratio_mean,
                   bed_bath_diff, bed_bath_ratio]

    big_arr = big[cols_to_use]
    small_arr = small[cols_to_use]
    target_vals = ['high', 'medium', 'low']
    for t in target_vals:
        big_target = big[TARGET].apply(lambda s: 1 if s == t else 0)
        small_target = small[TARGET].apply(lambda s: 1 if s == t else 0)
        model = LogisticRegression()
        model.fit(big_arr, big_target)
        proba = model.predict_proba(small_arr)[:, 1]
        auc = roc_auc_score(small_target, proba)
        loss = log_loss(small_target, proba)
        print 'auc={}'.format(auc)
        print 'log_loss={}'.format(loss)
        update_df.loc[small.index, 'log_reg_{}'.format(t)] = proba


def process_mngr_ens(train_df, test_df):
    col = MANAGER_ID
    folds = 5
    add_log_reg_cols(train_df, test_df, col, folds)
    return train_df, test_df, ['log_reg_{}'.format(t) for t in ['high', 'medium', 'low']]
