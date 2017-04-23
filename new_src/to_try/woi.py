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

def woi(good, bed, prior):
    if bed==0:
        return 2*math.log(good/(1*prior))
    if good==0:
        return 2*math.log(1/(bed*prior))
    return math.log(good/(bed*prior))

def process_woi(train_df, test_df, variable, folds):
    skf = StratifiedKFold(folds)
    prior = pd.get_dummies(train_df, columns=[TARGET])[['interest_level_high', 'interest_level_medium', 'interest_level_low']].mean()
    prior['high']=prior['interest_level_high']
    prior['medium']=prior['interest_level_medium']
    prior['low']=prior['interest_level_low']

    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df[TARGET]):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        process_woi_cols_big_small(big, small, prior, train_df, variable)

    process_woi_cols_big_small(train_df, test_df, prior, test_df, variable)



def process_woi_cols_big_small(big, small, prior, update_df, variable):
    # woi_f = np.vectorize(woi)
    big['t'] = big[TARGET]
    big = pd.get_dummies(big, columns=['t'])
    agg = OrderedDict([
        ('t_high', {'high': 'sum'}),
        ('t_medium', {'medium': 'sum'}),
        ('t_low', {'low': 'sum'})
    ])
    df = big.groupby(variable).agg(agg)
    cols = ['high', 'medium', 'low']
    df.columns = cols

    prior_high = prior['high'] / (prior['low'] + prior['medium'])
    prior_medium = prior['medium'] / (prior['low'] + prior['high'])
    prior_low = prior['low'] / (prior['high'] + prior['medium'])


    df['woi_high']= df.apply(lambda s: woi(s['high'],       s['low'] + s['medium'], prior_high), axis=1)
    df['woi_medium']= df.apply(lambda s: woi(s['medium'],   s['low'] + s['high'],  prior_medium), axis=1)
    df['woi_low']= df.apply(lambda s: woi(s['low'],       s['high'] + s['medium'],  prior_low), axis=1)

    # big = pd.merge(big, df, left_on=variable, right_index=True)
    for t in ['high', 'medium', 'low']:
        col = 'woi_{}'.format(t)
        if col in small:
            del small[col]

    small = pd.merge(small, df,left_on=variable, right_index=True, how='left')
    small.loc[small['high'].isnull(), cols] = 0


    target_vals = ['high', 'medium', 'low']
    for t in target_vals:
        update_df.loc[small.index, 'woi_{}'.format(t)] = small['woi_{}'.format(t)]


def process_mngr_woi(train_df, test_df):
    col = MANAGER_ID
    folds = 5
    process_woi(train_df, test_df, col, folds)
    return train_df, test_df, ['woi_{}'.format(t) for t in ['high', 'medium', 'low']]
