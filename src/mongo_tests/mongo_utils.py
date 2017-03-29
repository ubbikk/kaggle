from pymongo import MongoClient
import numpy as np
from numpy import mean, std
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sns

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

client = MongoClient('10.20.0.144', 27017)
db = client.renthop_results

def load_results(name):
    collection = db[name]
    return [x['results'] for x in collection.find()]

def load_results_1K(name):
    collection = db[name]
    return [x['results']['test'][1000] for x in collection.find()]

def load_importance(name, features):
    arr = load_importance_raw(name)
    sz = len(features)
    res = [np.mean([x[j] for x in arr]) for j in range(sz)]
    stds = [np.std([x[j] for x in arr]) for j in range(sz)]
    res = zip(features, res, stds)
    res.sort(key=lambda s: s[1], reverse=True)
    return res


def load_importance_raw(name):
    collection = db[name]
    return [x['importance'] for x in collection.find()]

def explore_importance(name, features, N):
    res = load_importance(name, features)
    res=res[:N]
    xs = [x[0] for x in res]
    ys=[x[1] for x in res]
    sns.barplot(xs, ys)
    sns.plt.show()


def explore_res_name(name):
    res = load_results_1K(name)
    return [
        ('avg_loss', np.mean(res)),
        ('max', np.max(res)),
        ('min', np.min(res)),
        ('std', np.std(res)),
        ('len', len(res))
    ]