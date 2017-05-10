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
CREATED = "created"
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
DAY_OF_YEAR='day_of_year'
LABEL = 'label'
SECONDS='seconds'


FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)
# pd.set_option('display.max_colwidth', 200)

train_file = '../data/redhoop/train.json'
test_file = '../data/redhoop/test.json'

def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]


def load_train():
    return basic_preprocess(pd.read_json(train_file))


def load_test():
    return basic_preprocess(pd.read_json(test_file))

def process_outliers_lat_long(train_df, test_df):
    min_lat=40
    max_lat=41
    min_long=-74.1
    max_long=-73

    good_lat = (train_df[LATITUDE] < max_lat) & (train_df[LATITUDE] > min_lat)
    good_long = (train_df[LONGITUDE] < max_long) & (train_df[LONGITUDE] > min_long)

    train_df = train_df[good_lat & good_long]

    bed_lat = (test_df[LATITUDE] >=max_lat) | (test_df[LATITUDE] <=min_lat)
    bed_long = (test_df[LONGITUDE] >= max_long) | (test_df[LONGITUDE] <= min_long)
    test_df[LATITUDE][bed_lat] = train_df[LATITUDE].mean()
    test_df[LONGITUDE][bed_long]=train_df[LONGITUDE].mean()

    return train_df, test_df


def basic_preprocess(df):
    df['num_features'] = df[u'features'].apply(len)
    df['num_photos'] = df['photos'].apply(len)
    df['word_num_in_descr'] = df['description'].apply(lambda x: len(x.split(' ')))
    df["created"] = pd.to_datetime(df["created"])
    # df["created_year"] = df["created"].dt.year
    df[CREATED_MONTH] = df["created"].dt.month
    df[CREATED_DAY] = df["created"].dt.day
    df[CREATED_HOUR] = df["created"].dt.hour
    df[CREATED_MINUTE] = df["created"].dt.minute
    df[DAY_OF_WEEK] = df['created'].dt.dayofweek
    df[DAY_OF_YEAR] = df['created'].dt.dayofyear
    df[SECONDS] = df['created'].dt.second
    bc_price, tmp = boxcox(df['price'])
    df['bc_price'] = bc_price


    return df


magic_file = '../data/redhoop/listing_image_time.csv'
LISTING_ID = 'listing_id'
TIME_STAMP='time_stamp'
MAGIC = "img_date"


def process_magic(train_df, test_df):
    image_date = pd.read_csv(magic_file)

    image_date.loc[80240,"time_stamp"] = 1478129766

    image_date["img_date"] = pd.to_datetime(image_date["time_stamp"], unit="s")
    image_date["img_days_passed"] = (image_date["img_date"].max() - image_date["img_date"]).astype(
        "timedelta64[D]").astype(int)
    image_date["img_date_month"] = image_date["img_date"].dt.month
    image_date["img_date_week"] = image_date["img_date"].dt.week
    image_date["img_date_day"] = image_date["img_date"].dt.day
    image_date["img_date_dayofweek"] = image_date["img_date"].dt.dayofweek
    image_date["img_date_dayofyear"] = image_date["img_date"].dt.dayofyear
    image_date["img_date_hour"] = image_date["img_date"].dt.hour
    image_date["img_date_minute"] = image_date["img_date"].dt.minute
    image_date["img_date_second"] = image_date["img_date"].dt.second
    image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(
        lambda x: 1 if x < 10 else 2 if x < 20 else 3)

    df = pd.concat([train_df, test_df])
    df = pd.merge(df, image_date,   left_on=LISTING_ID, right_on='Listing_Id')
    new_cols = ["img_days_passed","img_date_month","img_date_week",
                "img_date_day","img_date_dayofweek","img_date_dayofyear",
                "img_date_hour", "img_date_monthBeginMidEnd",
                "img_date_minute", "img_date_second", "img_date", "time_stamp"]

    # for col in new_cols:
    #     train_df[col] = df.loc[train_df.index, col]
    #     test_df[col] = df.loc[test_df.index, col]

    df_to_merge = df[[LISTING_ID] + new_cols]
    train_df = pd.merge(train_df, df_to_merge, on=LISTING_ID)
    test_df = pd.merge(test_df, df_to_merge, on=LISTING_ID)

    return train_df, test_df, new_cols

def get_group_by_mngr_target_dummies(df):
    df = pd.get_dummies(df, columns=[TARGET])
    target_vals = ['high', 'medium', 'low']
    dummies = ['interest_level_{}'.format(k) for k in target_vals]
    return df.groupby(MANAGER_ID)[dummies].sum()

def get_target_means_by_mngr(df):
    df = pd.get_dummies(df, columns=[TARGET])
    target_vals = ['high', 'medium', 'low']
    dummies = ['interest_level_{}'.format(k) for k in target_vals]
    means= df.groupby(MANAGER_ID)[dummies].mean()
    means['count'] = df.groupby(MANAGER_ID)[MANAGER_ID].count()
    return means.sort_values(by=['count'], ascending=False)

def explore_target(df):
    print 'high         {}'.format(len(df[df[TARGET]=='high'])/(1.0*len(df)))
    print 'medium       {}'.format(len(df[df[TARGET]=='medium'])/(1.0*len(df)))
    print 'low          {}'.format(len(df[df[TARGET]=='low'])/(1.0*len(df)))

def get_top_N_vals_of_column(df, col, N):
    bl=df.groupby(col)[col].count().sort_values(ascending=False)
    return bl.index[:N].values


def visualize_condit_counts_of_target_on_top_N_values_of_col(df, col, target_col, N=None):
    if N is None:
        N=len(set(df[col]))

    top_N = get_top_N_vals_of_column(df, col, N)
    df = df[df[col].apply(lambda s: s in top_N)]
    sns.factorplot(target_col, col=col, data=df, kind='count', estimator='mean')

train_df, test_df = load_train(), load_test()
train_df, test_df, magic_cols = process_magic(train_df, test_df)
