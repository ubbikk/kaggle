import json
import numpy as np
from hyperopt.mongoexp import MongoTrials
from scipy.stats import ttest_ind
from collections import OrderedDict
import seaborn as sns
import pandas as pd


def explore_res_fp(fp):
    res = json.load(open(fp))
    return [
        ('avg_loss', np.mean(res)),
        ('std', np.std(res)),
        ('len', len(res))
    ]

def explore_res_arr(res):
    return [
        ('avg_loss', np.mean(res)),
        ('std', np.std(res)),
        ('len', len(res))
    ]


def load_fp(fp):
    return json.load(open(fp))


def get_trials(key):
    return MongoTrials('mongo://10.20.0.144:27017/{}/jobs'.format(key), exp_key='exp1')

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

def load_importance(fp, features):
    arr = load_fp(fp)
    sz = len(features)
    res = [np.mean([x[j] for x in arr]) for j in range(sz)]
    stds = [np.std([x[j] for x in arr]) for j in range(sz)]
    res = zip(features, res, stds)
    res.sort(key=lambda s: s[1], reverse=True)
    return res

def explore_importance_old(res, N):
    res=res[:N]
    xs = [x[0] for x in res]
    ys=[x[1] for x in res]
    sns.barplot(xs, ys)
    sns.plt.show()


def explore_importance_fp(fp, features, N):
    res = load_importance(fp, features)
    res=res[:N]
    xs = [x[0] for x in res]
    ys=[x[1] for x in res]
    sns.barplot(xs, ys)
    sns.plt.show()

def explore_importance_and_std(fp, features, N):
    arr = load_fp(fp)
    features=features[:N]
    importances=[]
    ff=[]
    for run_i in range(len(arr)):
        for fi in range(N):
            importances.append(arr[run_i][fi])
            ff.append(features[fi])

    df = pd.DataFrame({'feature':ff, 'importance':importances})
    sns.barplot(x='feature', y='importance', data=df)
    sns.plt.show()