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
CREATED='created'
LABEL='lbl'




def process_mngr_target_ratios(train_df, test_df):
    N=10
    folds=5
    return process_target_ratios(train_df, test_df, MANAGER_ID, folds, N)



def process_target_ratios(train_df, test_df, col, folds, N):
    target_vals = ['high', 'medium', 'low']
    new_cols = {k: '{}_target_ratios_{}'.format(col, k) for k in target_vals}
    results=[]
    for _ in range(N):
        random_state = int(time())
        skf = StratifiedKFold(folds, random_state=random_state, shuffle=True)
        print random_state

        update_df = train_df[PRICE].to_frame('bl')
        results.append(update_df)

        for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
            big = train_df.iloc[big_ind]
            small = train_df.iloc[small_ind]
            calc_target_ratios(big, small, col, new_cols, update_df)

    for x in new_cols.values():
        train_df[x]=np.mean([x[x] for x in results])

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

