from hyperopt.mongoexp import MongoTrials


def get_trials(key):
    return MongoTrials('mongo://10.20.0.144:27017/{}/jobs'.format(key), exp_key='exp1')

def get_best_loss(trials):
    return trials.best_trial['result']['loss']

def get_vals(trials):
    return trials.best_trial['misc']['vals']
