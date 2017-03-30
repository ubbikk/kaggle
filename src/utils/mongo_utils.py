from pymongo import MongoClient
import numpy as np
from numpy import mean, std
from scipy.stats import ttest_ind, normaltest
import pandas as pd
import seaborn as sns
from hyperopt.mongoexp import MongoTrials
import math
import matplotlib.pyplot as plt

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

gc_host='35.187.46.132'
local_host = '10.20.0.144'
client = MongoClient(gc_host, 27017)
db = client.renthop_results




def get_trials(key, host=gc_host):
    return MongoTrials('mongo://{}:27017/{}/jobs'.format(host, key), exp_key='exp1')

def get_best_loss(trials):
    return trials.best_trial['result']['loss']

def get_vals(t):
    return t['misc']['vals']

def get_params(t):
    vals= get_vals(t)
    vals = dict(vals)
    return {k:v[0] for k,v in vals.iteritems()}

def get_loss(t):
    if 'loss' in t['result']:
        return t['result']['loss']
    return None

def get_best_params(trials, num):
    trials = filter(lambda t: get_loss(t) is not None, trials)
    s = [(get_loss(t), get_params(t)) for t in trials]
    s.sort(key=lambda x:x[0])

    return s[:num]

def load_results(name):
    collection = db[name]
    return [x['results'] for x in collection.find()]

def load_results_1K(name):
    collection = db[name]
    return [x['results']['test'][1000] for x in collection.find()]

def load_results_atN(name, N):
    collection = db[name]
    return [x['results']['test'][N] for x in collection.find()]

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
    s = std(res)
    m = mean(res)
    return [
        ('avg_loss  ', m),
        ('max       ', max(res)),
        ('min       ', min(res)),
        ('2s        ', '[{}, {}]'.format(m-2*s, m+2*s)),
        ('3s        ', '[{}, {}]'.format(m-3*s, m+3*s)),
        ('norm      ', normaltest(res).pvalue),
        ('std       ', s),
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