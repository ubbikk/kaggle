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

def hcc_encode(train_df, test_df, variable, binary_target, k=5, f=1, g=1, r_k=0.01, folds=5):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    prior_prob = train_df[binary_target].mean()
    hcc_name = "_".join(["hcc", variable, binary_target])

    skf = StratifiedKFold(folds)
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        grouped = big.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
        grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
        grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

        if hcc_name in small.columns:
            del small[hcc_name]
        small = pd.merge(small, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
        small.loc[small[hcc_name].isnull(), hcc_name] = prior_prob
        small[hcc_name] = small[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(small))
        train_df.loc[small.index, hcc_name] = small[hcc_name]

    grouped = train_df.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    test_df = pd.merge(test_df, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
    test_df.loc[test_df[hcc_name].isnull(), hcc_name] = prior_prob

    return train_df, test_df, hcc_name


def process_mngr_ens(train_df, test_df, target_val):
    model = LogisticRegression()
    train_df['t'] = train_df[TARGET].apply(lambda s: 1 if s==target_val else 0)
    df = pd.get_dummies(train_df, columns=[TARGET])
