from hyperopt.mongoexp import MongoTrials


def get_trials():
    return MongoTrials('mongo://10.20.0.144:27017/redhop_mngr_id_exp_family/jobs', exp_key='exp1')


