import numpy as np
import pandas as pd

CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
CREATED = 'created'
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'
NEI = 'neighbourhood'
BORO = 'boro'

def process_time_density_neis_df(df, ns_vals, periods):
    new_cols=[]
    for nei_col in [NEI_1, NEI_2]:
        tmp=None
        nei_vals = set(df[nei_col])
        for nei in nei_vals:
            bl=df[df[nei_col]==nei]
            bl, tmp=process_time_density_df(bl, ns_vals, periods, nei_col)
            df.loc[bl.index] = bl
        new_cols+=tmp

    return df, new_cols

def process_time_density_df(df, ns_vals, periods, suffix):
    m = df[CREATED].sort_values().to_dict()
    m = [(k,v) for k,v in m.iteritems()]
    m.sort(key=lambda s: s[1])
    m = [(x[0], x[1].to_datetime()) for x in m]
    start = m[0][1]
    m = [(x[0], (x[1]-start).total_seconds()) for x in m]


    max_period = max(periods)
    sz = len(m)
    new_cols = []

    periods_data=[]
    for j in range(sz):
        ppd = {x:np.nan for x in periods}
        periods_data.append(ppd)
        for i in range(j+1, sz):
            delta = m[i][1]-m[j][1]
            if delta>max_period:
                break
            for p in periods:
                if delta<=p:
                    if np.isnan(ppd[p]):
                        ppd[p]=1
                    else:
                        ppd[p]+=1

    nns_data =[]
    for j in range(sz):
        nns = {x:np.nan for x in  ns_vals}
        nns_data.append(nns)
        for n in ns_vals:
            i = j+n
            if i>= sz:
                break

            v = m[i][1]-m[j][1]
            nns[n]=v


    indexes = [x[0] for x in m]

    norm = sz/1000.0

    for p in periods:
        vals = [x[p] for x in periods_data]
        col_name = '{}_num_in_period_{}'.format(suffix,p)
        new_cols.append(col_name)
        s= pd.Series({indexes[j]: vals[j] for j in range(sz)})
        # s=s/norm
        s=s.to_frame(col_name)
        df = pd.merge(df, s, left_index=True, right_index=True)

    for n in ns_vals:
        vals = [x[n] for x in nns_data]
        col_name = '{}_secs_until_{}'.format(suffix,n)
        new_cols.append(col_name)
        s= pd.Series({indexes[j]: vals[j] for j in range(sz)})
        # s=s/norm
        s=s.to_frame(col_name)
        df = pd.merge(df, s, left_index=True, right_index=True)

    return df, new_cols


def process_time_density(train_df, test_df):
    m=60.0
    h= 60*m
    d = 24*h
    periods=[m, 5*m, 10*m, 30*m, h, 3*h, 6*h, 12*h, d, 3*d, 7*d, 30*d]
    nums=[1, 2, 3, 5, 10, 50, 100, 300, 1000]

    train_df, new_cols = process_time_density_df(train_df, nums, periods)
    train_df['label']='train'
    test_df['label'] = 'test'

    train_df_copy = train_df.copy()
    for col in new_cols:
        del train_df_copy[col]

    df = pd.concat([train_df_copy, test_df])
    df, new_cols = process_time_density_df(df, nums, periods)
    test_df=df[df['label']=='test']

    return train_df, test_df, new_cols






