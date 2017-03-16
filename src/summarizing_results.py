import json
import numpy as np
from hyperopt.mongoexp import MongoTrials
from scipy.stats import ttest_ind
from collections import OrderedDict


def explore_res(fp):
    res = json.load(open(fp))
    return [
        ('loss', np.mean(res)),
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