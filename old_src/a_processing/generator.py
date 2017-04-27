import json
import os
import traceback
from time import time, sleep

import seaborn as sns
import pandas as pd
from collections import OrderedDict

import sys
from matplotlib import pyplot
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.spatial import KDTree
import math
from pymongo import MongoClient

# seeds_fp = '../../seeds.json'
# splits_big_fp='../../splits_big.json'
# splits_small_fp='../../splits_small.json'

seeds_fp = '../seeds.json'
splits_big_fp='../splits_big.json'
splits_small_fp='../splits_small.json'

SEEDS = json.load(open(seeds_fp))
SPLITS_BIG=json.load(open(splits_big_fp))
SPLITS_SMALL=json.load(open(splits_small_fp))


def getN(mongo_host, name, experiment_max_time):
    client = MongoClient(mongo_host, 27017)
    db = client[name]
    collection = db['splits_control'.format(name)]
    res = [x for x in collection.find()]

    res.sort(key=lambda s: s['N'])

    for con in res:
        if (not con['finished']) and (time()-con['time'] > experiment_max_time):
            N = con['N']
            collection.replace_one({'N': N}, {'N': N, 'time': time(), 'finished': False})
            return N

    N = len(res)
    collection.insert_one({'N': N, 'time': time(), 'finished': False})

    return N

def split_from_N(df, N):
    return df.loc[SPLITS_BIG[N],:], df.loc[SPLITS_SMALL[N], :]


def complete_split_mongo(N, name, mongo_host, probs, test_indexes, losses, importance, f_names):
    client = MongoClient(mongo_host, 27017)
    db = client[name]

    collection = db['probs']
    collection.insert_one({'N': N, 'val': probs, 'index':test_indexes})

    collection = db['losses']
    collection.insert_one({'N': N, 'val': losses})

    collection = db['importance']
    collection.insert_one({'N': N, 'val': importance})

    collection = db['features']
    collection.insert_one({'N': N, 'val': f_names})

    collection = db['splits_control'.format(name)]
    collection.replace_one({'N': N}, {'N': N, 'time': time(), 'finished': True})


def get_probs_from_est(estimator, proba, test_df):
    classes = [x for x in estimator.classes_]
    res = {}
    for cl in classes:
        p=proba[:, classes.index(cl)]
        res[cl] = [a.item() for a in p]
    return res, [x for x in test_df.index.values]


def complete_split_file(ii, l, name):
    fp = name + '_results.json'
    ii_fp = name + '_importance.json'
    with open(fp, 'w+') as f:
        json.dump(l, f)
    with open(ii_fp, 'w+') as f:
        json.dump(ii, f)
