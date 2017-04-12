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

def out(l, loss, l_1K, loss1K, num, t):
    print '\n\n'
    print '#{}'.format(num)
    if loss1K is not None:
        print 'loss1K {}'.format(loss1K)
        print 'avg_loss1K {}'.format(np.mean(l_1K))
        print get_3s_confidence_for_mean(l_1K)
        print

    print 'loss {}'.format(loss)
    print 'avg_loss {}'.format(np.mean(l))
    print get_3s_confidence_for_mean(l)
    print 'std {}'.format(np.std(l))
    print 'time {}'.format(t)

def get_3s_confidence_for_mean(l):
    std = np.std(l)/math.sqrt(len(l))
    m = np.mean(l)
    start = m -3*std
    end = m+3*std

    return '3s_confidence: [{}, {}]'.format(start, end)

def write_results(l, fp):
    with open(fp, 'w+') as f:
        json.dump(l, f)


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


def get_loss_at1K(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    return results_on_test[1000]


BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'

def normalize_bed_bath_good(df):
    df['bed_bath']=df[[BEDROOMS, BATHROOMS]].apply(lambda s: (s[BEDROOMS], s[BATHROOMS]), axis=1)
    def norm(s):
        bed=s[0]
        bath=s[1]
        if bed==0:
            if bath>=1.5:
                return [0,2.0]
        elif bed==1:
            if bath>=2.5:
                return [1,2.0]
        elif bed==2:
            if bath>=3.0:
                return [2,3.0]
        elif bed==3:
            if bath>=4.0:
                return [3,4.0]
        elif bed==4:
            if bath==0:
                return [4,1]
            elif bath>=4.5:
                return [4,4.5]
        elif bed>=5:
            if bath <=1.5:
                return [5,1.5]
            elif bath <=2.5:
                return [5,2.5]
            elif bath <=3.5:
                return [5,3]
            else:
                return [5,4]

        return [bed, bath]

    df['bed_bath']=df['bed_bath'].apply(norm)
    df[BED_NORMALIZED]=df['bed_bath'].apply(lambda s:s[0])
    df[BATH_NORMALIZED]=df['bed_bath'].apply(lambda s:s[1])

def explore_target(df):
    print 'high         {}'.format(len(df[df[TARGET]=='high'])/(1.0*len(df)))
    print 'medium       {}'.format(len(df[df[TARGET]=='medium'])/(1.0*len(df)))
    print 'low          {}'.format(len(df[df[TARGET]=='low'])/(1.0*len(df)))

train_df, test_df = load_train(), load_test()
normalize_bed_bath_good(train_df)
normalize_bed_bath_good(test_df)
df = train_df
price_median = 'price_median'
df[price_median]=df.groupby([BED_NORMALIZED, BATH_NORMALIZED])[PRICE].transform('median')
median_ratio = 'median_ratio'
df[median_ratio] = (df[PRICE]-df[price_median])/df[price_median]
