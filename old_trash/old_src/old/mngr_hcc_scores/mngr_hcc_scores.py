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


def process_mngr_hcc_scores(train_df, test_df):
    mngr_hcc_high = 'hcc_manager_id_target_high'
    mngr_hcc_medium = 'hcc_manager_id_target_medium'
    mngr_hcc_skill = 'mngr_hcc_skill'

    mngr_hcc_cols = mngr_hcc_high, mngr_hcc_medium, mngr_hcc_skill

    df = pd.concat([train_df, test_df])
    df[mngr_hcc_skill]=df[mngr_hcc_high]+3*df[mngr_hcc_medium]


    new_cols =[]
    for col in mngr_hcc_cols:
        new_col = 'rank_of_{}'.format(col)
        df[new_col] = df[col].rank(method='dense')
        new_cols.append(new_col)

        new_col = 'rank_of_{}_by_nei2'.format(col)
        df[new_col]=df.groupby(NEI_2)[col].rank(method='dense')
        new_cols.append(new_col)

        new_col = 'rank_of_{}_by_nei1'.format(col)
        df[new_col]=df.groupby(NEI_1)[col].rank(method='dense')
        new_cols.append(new_col)

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols



