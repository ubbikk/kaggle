import numpy as np
import pandas as pd

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
CREATED='created'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'

def process_recent_mngr_activity_df(df, periods):
    bl=df.groupby(MANAGER_ID).groups
    prev_cols = {p:'mngr_activity_prev_num_in_{}_secs'.format(p) for p in periods}
    next_cols = {p:'mngr_activity_next_num_in_{}_secs'.format(p) for p in periods}
    count=0
    mngrs_num = len(bl)
    prev_all=[]
    next_all=[]
    index_all=[]
    for m, ii in bl.iteritems():
        s = df.loc[ii][CREATED].sort_values()
        sz = len(s)
        ind = list(s.index.values)
        index_all+=ind
        vals = s.values
        start = vals[0]
        max_period = max(periods)
        vals = [(x-start)/np.timedelta64(1, 's') for x in vals]
        prev = {p:[0 for x in range(sz)] for p in periods}
        prev_all.append(prev)
        nnext = {p:[0 for x in range(sz)] for p in periods}
        next_all.append(nnext)

        for j in range(sz):
            for i in range(j+1, sz):
                delta = vals[i] - vals[j]
                if delta > max_period:
                    break
                for p in periods:
                    if delta<=p:
                        nnext[p][j]=1

            for i in range(j-1, -1, -1):
                delta = vals[j] - vals[i]
                if delta > max_period:
                    break
                for p in periods:
                    if delta<=p:
                        prev[p][j]=1

    for p in periods:
        prev_col = prev_cols[p]
        bl = [x[p] for x in prev_all]
        vals =[]
        for x in bl:
            vals+=x
        df.loc[index_all, prev_col] = vals

        next_col = next_cols[p]
        bl = [x[p] for x in next_all]
        vals =[]
        for x in bl:
            vals+=x
        df.loc[index_all, next_col] = vals

    new_cols = prev_cols.values()+next_cols.values()

    return df, new_cols


def process_recent_mngr_activity(train_df, test_df):
    m=60.0
    h= 60*m
    d = 24*h
    periods=[d+m, h, 30*m]

    train_df, new_cols = process_recent_mngr_activity_df(train_df, periods)
    train_df['label']='train'
    test_df['label'] = 'test'

    train_df_copy = train_df.copy()
    for col in new_cols:
        del train_df_copy[col]

    df = pd.concat([train_df_copy, test_df])
    df, new_cols = process_recent_mngr_activity_df(df, periods)
    test_df=df[df['label']=='test']

    return train_df, test_df, new_cols



