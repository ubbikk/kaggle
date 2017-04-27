import json
import math
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hyperopt.mongoexp import MongoTrials
from numpy import mean, std
from pymongo import MongoClient
from scipy.stats import normaltest
import pandas as pd
from sklearn.metrics import log_loss
import os

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

gc_host = '35.187.46.132'
local_host = '10.20.0.144'

host = gc_host

client = MongoClient(host, 27017)
CV = 5
TARGET = 'interest_level'
LISTING_ID = 'listing_id'



######################################################3
#VALIDATION
######################################################3

stacking_fp = '../stacking_data'
splits_small_fp='../splits_small.json'
splits_big_fp='../splits_big.json'
SPLITS_SMALL=json.load(open(splits_small_fp))[:5]
SPLITS_BIG=json.load(open(splits_big_fp))



def load_from_db_and_store_avg_validation_df(name, fp=None):
    probs = get_all_probs(name)
    if fp is None:
        fp = os.path.join(stacking_fp, '{}_{}.csv'.format(name, len(probs)))
    df = sum(probs)/len(probs)
    df['fold'] = df['fold'].astype(np.int64)
    df.to_csv(fp, index_label=LISTING_ID)

def load_from_db_avg_validation_df(name):
    probs = get_all_probs(name)
    df = sum(probs)/len(probs)
    df['fold'] = df['fold'].astype(np.int64)

    return df


def get_fold_index(i):
    if isinstance(i, np.int64):
        i=i.item()
    elif isinstance(i, str):
        i=int(i)
    for j in range(CV):
        if i in SPLITS_SMALL[j]:
            return j

    raise


def get_one_cv__probs(name, n=0):
    probs = []
    ns = [CV * n + j for j in range(CV)]
    db = client[name]
    collection = db['probs']
    for p in collection.find():
        N = p['N']
        if N in ns:
            probs.append(p)
        if len(probs) == CV:
            break

    if len(probs) < CV:
        raise

    return convert_5_cv_entries_to_df(probs)


def convert_5_cv_entries_to_df(probs):
    dfs = []
    for p in probs:
        df = pd.DataFrame(p['val'], index=p['index'])
        fold = get_fold_index(df.index.values[0])
        df['fold'] = fold
        dfs.append(df)
    return pd.concat(dfs)[['low', 'medium', 'high', 'fold']]


def get_all_probs_very_raw(name):
    db = client[name]
    collection = db['probs']
    return [p for p in collection.find()]


def get_all_probs(name):
    res = get_all_probs_raw(name)
    return [convert_5_cv_entries_to_df(p) for p in res]


def get_all_probs_raw(name):
    probs_map = {}
    db = client[name]
    collection = db['probs']
    for p in collection.find():
        N = p['N']
        probs_map[N] = p

    m = (1 + max(probs_map.keys())) / CV
    res = []
    for j in range(m):
        probs = []
        for i in range(CV * j, CV * (j + 1)):
            if i in probs_map:
                probs.append(probs_map[i])

        if len(probs) != CV:
            continue
        res.append(probs)

    return res

def df_form_probs_mg_entry(p):
    return pd.DataFrame(p['val'], index=p['index'])[['high', 'low', 'medium']]

def explore_stack_cv_error(probs_raw, train_df):
    cv_dfs =[[] for _ in range(CV)]
    indexes = [probs_raw[0][j]['index'] for j in range(CV)]
    for cv in probs_raw:
        for j in range(CV):
            df = df_form_probs_mg_entry(cv[j])
            cv_dfs[j].append(df)

    cv_dfs = [sum(x)/len(x) for x in cv_dfs]

    cv_loses = [log_loss(train_df[TARGET].loc[indexes[j]], cv_dfs[j].loc[indexes[j]].values) for j in range(CV)]

    return {
        'all':cv_loses,
        'avg': np.mean(cv_loses)
    }


def explore_cv_errors_name(name, train_df):
    probs_raw = get_all_probs_raw(name)
    return explore_cv_errors(probs_raw, train_df)


def explore_cv_errors(probs_raw, train_df):
    cv_errors = []
    errors_flat = []
    for p in probs_raw:
        errors = []
        for f in p:
            index = f['index']
            target = train_df.loc[index][TARGET]
            df = pd.DataFrame(f['val'])
            probs = df[['high', 'low', 'medium']].values
            loss = log_loss(target, probs)
            errors.append(loss)
            errors_flat.append(loss)

        cv_errors.append(np.mean(errors))

    return [
        ('num', len(probs_raw)),
        ('cv_mean', np.mean(cv_errors)),
        ('cv_std', np.std(cv_errors)),
        ('cv_max', np.max(cv_errors)),
        ('cv_min', np.min(cv_errors)),

        ('flat_mean', np.mean(errors_flat)),
        ('flat_std', np.std(errors_flat)),
        ('flat_max', np.max(errors_flat)),
        ('flat_min', np.min(errors_flat)), ]


def plot_errors_new(name):
    db = client[name]
    results = db['losses']
    results = [x['val'] for x in results.find()]
    train_runs= [x['train'] for x in results]
    test_runs= [x['test'] for x in results]

    sz=len(train_runs[0])
    x_axis=range(sz)
    y_train = [np.mean([x[j] for x in train_runs]) for j in x_axis]
    y_test = [np.mean([x[j] for x in test_runs]) for j in x_axis]

    fig, ax = plt.subplots()
    ax.plot(x_axis, y_train, label='train')
    ax.plot(x_axis, y_test, label='test')
    ax.legend()


def load_importance_raw(name):
    db = client[name]
    collection = db['importance']
    return [x['val'] for x in collection.find()]

def load_importance(name, features):
    arr = load_importance_raw(name)
    sz = len(features)
    res = [np.mean([x[j] for x in arr]) for j in range(sz)]
    stds = [np.std([x[j] for x in arr]) for j in range(sz)]
    res = zip(features, res, stds)
    res.sort(key=lambda s: s[1], reverse=True)
    return res

def explore_importance_new(name, features, N=None):
    if N is None:
        N=len(features)

    res = load_importance(name, features)
    print res
    res=res[:N]
    xs = [x[0] for x in res]
    ys=[x[1] for x in res]
    sns.barplot(xs, ys)
    sns.plt.show()


def store_experiments(experiments):
    for e in experiments:
        load_from_db_and_store_avg_validation_df(e)

######################################################3
#VALIDATION
######################################################3



#######################################################
#Results
#######################################################
stacking_submit_fp = '../stacking_submit_data'


def store_submits(experiments):
    for e in experiments:
        load_from_db_and_store_avg_submit_df(e)

def submit_item_to_df(res):
    return pd.DataFrame({k: res[k] for k in ['high', 'low', 'medium']}, index=res[LISTING_ID])


# def create_avg_submit(name):
#     probs = load_from_db_submit_raw(name)
#     create_avg_submit_from_probs(probs, name)
#
#
# def create_avg_submit_from_probs(probs, name):
#     print 'len = {}'.format(len(probs))
#
#     df = sum(probs) / len(probs)
#     df[LISTING_ID] = df.index.values
#     df = df[[LISTING_ID, 'high', 'medium', 'low']]
#     fp = '{}_results_{}.csv'.format(name, int(time()))
#     df.to_csv(fp, index=False)


def load_from_db_and_store_avg_submit_df(name, fp=None):
    probs = load_from_db_avg_submit(name)
    print '{}_{}'.format(name, len(probs))
    if fp is None:
        fp = os.path.join(stacking_submit_fp, '{}_{}.csv'.format(name, len(probs)))

    df = sum(probs)/len(probs)
    df.to_csv(fp, index_label=LISTING_ID)

def load_from_db_avg_submit(name):
    results = load_from_db_submit_raw(name)
    results = [submit_item_to_df(x) for x in results]
    return results

def load_from_db_submit_raw(name):
    db = client[name]
    collection = db['results']
    return [x for x in collection.find()]



#######################################################
#Results
#######################################################

