import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import boxcox
from sklearn.metrics import log_loss

src_folder = '/home/ubik/PycharmProjects/kaggle/src'
os.chdir(src_folder)
import sys

sys.path.append(src_folder)
#

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

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

# sns.set(color_codes=True)
# sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)

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
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    bc_price, tmp = boxcox(df['price'])
    df['bc_price'] = bc_price

    return df


# (0.61509489625789615, [0.61124170916042475, 0.61371758902339113, 0.61794752159334343, 0.61555861194203254, 0.61700904957028924])
def simple_loss(df):
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]

    train_df, test_df = split_df(df, 0.7)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values

    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=5000, objective='mlogloss')
    # estimator = RandomForestClassifier(n_estimators=1000)
    eval_set = [(train_arr, train_target), (test_arr, test_target)]
    estimator.fit(train_arr, train_target, eval_set=eval_set, eval_metric='mlogloss', early_stopping_rounds=100)

    proba = estimator.predict_proba(test_arr)
    loss = log_loss(test_target, proba)
    print loss
    plot_errors(estimator.evals_result())
    return loss

def plot_errors(results):
    fig, ax = plt.subplots()
    to_plot0 = results['validation_0']['mlogloss']
    to_plot1 = results['validation_1']['mlogloss']
    ax.plot(range(len(to_plot0)), to_plot0, label='train')
    ax.plot(range(len(to_plot0)), to_plot1, label='test')
    ax.legend()


def do_test(num, fp):
    neww = []
    for x in range(num):
        df = load_train()
        loss = simple_loss(df)
        print loss
        neww.append(loss)
        with open(fp, 'w+') as f:
            json.dump(neww, f)

    print '\n\n\n\n'
    print 'avg = {}'.format(np.mean(neww))

def test():
    df =load_train()
    loss = simple_loss(df)
    print loss

# test()



# train_df, test_df = load_train(), load_test()
