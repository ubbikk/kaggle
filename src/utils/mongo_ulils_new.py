import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hyperopt.mongoexp import MongoTrials
from numpy import mean, std
from pymongo import MongoClient
from scipy.stats import normaltest
import pandas as pd

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

gc_host='35.187.46.132'
local_host = '10.20.0.144'

host = gc_host

client = MongoClient(host, 27017)
CV=5


def load_results(name):
    db=client[name]
    collection = db['losses']
    return [x['val'] for x in collection.find()]

def load_results_1K(name):
    db=client[name]
    collection = db['losses']
    return [x['val']['test'][1000] for x in collection.find()]

def load_results_atN(name, N):
    db=client[name]
    collection = db['losses']
    return [x['val']['test'][N] for x in collection.find()]

def load_features_names(name):
    db=client[name]
    collection = db['features']
    for x in collection.find():
        return x['val']

def load_importance(name):
    features = load_features_names(name)
    sz = len(features)
    arr = load_importance_raw(name)
    res = [np.mean([x[j] for x in arr]) for j in range(sz)]
    stds = [np.std([x[j] for x in arr]) for j in range(sz)]
    res = zip(features, res, stds)
    res.sort(key=lambda s: s[1], reverse=True)
    return res


def load_importance_raw(name):
    db=client[name]
    collection = db['importance']
    return [x['val'] for x in collection.find()]

def explore_importance(name, N=None):
    features = load_features_names(name)
    if N is None:
        N=len(features)

    res = load_importance(name)
    print res
    res=res[:N]
    xs = [x[0] for x in res]
    ys=[x[1] for x in res]
    sns.barplot(xs, ys)
    sns.plt.show()


def explore_res_name(name):
    res = load_results_1K(name)
    s = std(res)/math.sqrt(len(res))
    m = mean(res)
    return [
        ('avg_loss  ', m),
        ('max       ', max(res)),
        ('min       ', min(res)),
        ('2s        ', '[{}, {}]'.format(m-2*s, m+2*s)),
        ('3s        ', '[{}, {}]'.format(m-3*s, m+3*s)),
        ('norm      ', normaltest(res).pvalue),
        ('std       ', np.std(res)),
        ('mean_std  ', s),
        ('len       ', len(res))
    ]


def plot_errors_name(name):
    results = load_results(name)
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


def get_probs(name, n=0):
    probs=[]
    ns=[CV*n + j for j in range(CV)]
    db=client[name]
    collection = db['probs']
    for p in collection.find():
        N = p['N']
        if N in ns:
            probs.append(p)
        if len(probs)==CV:
            break

    if len(probs)<CV:
        raise

    dfs=[]

    for p in probs:
        df = pd.DataFrame(p['val'], index=p['index'])
        dfs.append(df)

    return pd.concat(dfs)[['low', 'medium', 'high']]
