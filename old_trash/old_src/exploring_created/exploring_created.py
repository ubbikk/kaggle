import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
DAY_OF_YEAR='day_of_year'
CREATED='created'

def explore(df):
    df['m']=df[CREATED_MINUTE]+60*df[CREATED_HOUR]

def plot_m_count(df):
    df['m']=df[CREATED_MINUTE]+60*df[CREATED_HOUR]#+24*60*df[DAY_OF_YEAR]
    bl=df.groupby('m')['m'].count()
    fig, ax = plt.subplots()
    ax.plot(bl.index.values, bl.values, label='m-count')
    ax.legend()

def plot_target(df):
    h='interest_level_high'
    m='interest_level_medium'
    l='interest_level_low'
    df = df[(df[CREATED_HOUR]<8)&(df[CREATED_HOUR]>0)]
    df['m']=df[CREATED_MINUTE]+60*df[CREATED_HOUR]
    df = pd.get_dummies(df, columns=[TARGET])

    big_m= df.groupby('m')['m'].count()
    big_m = big_m[big_m>30].index.values


    bl=df.groupby('m')[[h,m,l]].mean().loc[big_m]
    counts =df.groupby('m')[h].count().loc[big_m].values

    fig, ax = plt.subplots()
    ax.plot(bl.index.values, bl[m].values, label='medium')
    ax.plot(bl.index.values, [blja(x) for x in counts], label='random')
    ax.legend()


def blja(n):
    h=0.07778813421948452
    m=0.22752877289674178
    return sum([(1 if np.random.random()<=m else 0) for x in range(n)])/float(n)


def plot_m_count_train_test(train_df, test_df):
    train_df['m']=train_df[CREATED_MINUTE]+60*train_df[CREATED_HOUR]
    test_df['m']=test_df[CREATED_MINUTE]+60*test_df[CREATED_HOUR]
    bl_train=train_df.groupby('m')['m'].count()
    bl_test=test_df.groupby('m')['m'].count()
    fig, ax = plt.subplots()
    normalizer = (float(len(test_df)) / len(train_df))
    ax.plot(bl_train.index.values, bl_train.values * normalizer, label='train')
    ax.plot(bl_test.index.values, bl_test.values, label='test')

    ax.legend()
