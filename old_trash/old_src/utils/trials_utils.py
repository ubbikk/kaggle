from hyperopt.mongoexp import MongoTrials
import math

gc_host='35.187.46.132'

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


