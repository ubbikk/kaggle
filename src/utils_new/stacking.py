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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)

CV = 5
TARGET = 'interest_level'
LISTING_ID = 'listing_id'

stacking_fp = '../stacking_data'
stacking_submit_fp = '../stacking_submit_data'
splits_small_fp = '../splits_small.json'
splits_big_fp = '../splits_big.json'
SPLITS_SMALL = json.load(open(splits_small_fp))[:5]
SPLITS_BIG = json.load(open(splits_big_fp))


def run_log_reg_cv(experiments, train_df, folder=stacking_fp):
    data = create_data_for_running_fs(experiments, train_df, folder)
    losses = []
    for train, test, train_target, test_target in data:
        model = LogisticRegression()
        model.fit(train.values, train_target)
        proba = model.predict_proba(test.values)
        loss = log_loss(test_target, proba)
        print loss
        losses.append(loss)

    return [
        ('avg', np.mean(losses)),
        ('losses', losses)
    ]

# [('avg', 0.51740182794744904),
#  ('losses',
#   [0.5156275620334051,
#    0.52082806399913439,
#    0.51685190142485371,
#    0.51170957753207724,
#    0.52199203474777456])]

# [('avg', 0.51655722401313642),
#  ('losses',
#   [0.51357636065967682,
#    0.51970377131222767,
#    0.51620198486410518,
#    0.51145152959603435,
#    0.52185247363363862])]


def run_xgb_cv(experiments, train_df, folder=stacking_fp):
    data = create_data_for_running_fs(experiments, train_df, folder)
    losses = []
    run_results = []
    for train, test, train_target, test_target in data:
        eval_set = [(train.values, train_target), (test.values, test_target)]
        model = xgb.XGBClassifier(objective='multi:softprob')
        model.fit(train.values, train_target, eval_set=eval_set, eval_metric='mlogloss', verbose=False)
        proba = model.predict_proba(test.values)
        loss = log_loss(test_target, proba)
        print loss
        losses.append(loss)
        run_results.append(xgboost_per_tree_results(model))

    # plot_errors_xgboost(run_results)

    return [
        ('avg', np.mean(losses)),
        ('losses', losses)
    ]


def run_rand_forest_cv(experiments, train_df, folder=stacking_fp):
    data = create_data_for_running_fs(experiments, train_df, folder)
    losses = []
    run_results = []
    for train, test, train_target, test_target in data:
        eval_set = [(train.values, train_target), (test.values, test_target)]
        model = RandomForestClassifier(n_estimators=100)
        model.fit(train.values, train_target)
        proba = model.predict_proba(test.values)
        loss = log_loss(test_target, proba)
        print loss
        losses.append(loss)
        # run_results.append(xgboost_per_tree_results(model))

    # plot_errors_xgboost(run_results)

    return [
        ('avg', np.mean(losses)),
        ('losses', losses)
    ]


def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    results_on_train = estimator.evals_result()['validation_0']['mlogloss']
    return {
        'train': results_on_train,
        'test': results_on_test
    }

def plot_errors_xgboost(results):
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

def avg_experiments(experiments, train_df, folder=stacking_fp):
    dfs = [(e, load_from_fs_avg_validation_df(e, folder)) for e in experiments]
    dfs=[x[1] for x in dfs]
    df = sum(dfs)/len(dfs)
    losses = []
    for cv in range(CV):
        index = SPLITS_SMALL[cv]
        proba = df.loc[index][['high', 'low', 'medium']]
        target = train_df.loc[index][TARGET]
        loss = log_loss(target, proba)
        losses.append(loss)

    return [
        ('avg', np.mean(losses)),
        ('losses', losses)
    ]


def load_from_fs_avg_validation_df(name, folder=None):
    if folder is None:
        folder = stacking_fp

    for f in os.listdir(folder):
        if f.startswith(name):
            return pd.read_csv(os.path.join(folder, f), index_col=LISTING_ID)

    print name
    raise


def load_and_unite_expiriments_fs(experiments, folder=stacking_fp):
    dfs = [(e, load_from_fs_avg_validation_df(e, folder)) for e in experiments]
    targets = ['low', 'medium', 'high']
    res_df = None
    counter = 0
    for e, df in dfs:
        if counter == 0:
            res_df = df[targets]
            res_df = res_df.rename(columns={k: '{}_{}'.format(e, k) for k in targets})
        else:
            tmp = df[targets]
            tmp = tmp.rename(columns={k: '{}_{}'.format(e, k) for k in targets})
            res_df = pd.merge(res_df, tmp, left_index=True, right_index=True)

        counter += 1

    return res_df


def create_data_for_running_fs(experiments, train_df, folder=stacking_fp):
    df = load_and_unite_expiriments_fs(experiments, folder)
    res = []
    for cv in range(CV):
        small_indexes = SPLITS_SMALL[cv]
        big_indexes = SPLITS_BIG[cv]
        train = df.loc[big_indexes]
        train_target = train_df.loc[big_indexes][TARGET]
        test_target = train_df.loc[small_indexes][TARGET]
        test = df.loc[small_indexes]
        res.append((train, test, train_target, test_target))

    return res


###################################################
#SUBMITTING....
###################################################
def submit_xgb(experiments, train_df, test_df,
               stacking_fldr=stacking_fp,
               submit_fldr=stacking_submit_fp):
    train = load_and_unite_expiriments_fs(experiments, stacking_fldr)
    test = load_and_unite_submits_fs(experiments, submit_fldr)

    features = train.columns.values

    train[TARGET] = train_df[TARGET]
    train_arr, train_target = train[features], train[TARGET]
    test_arr = test[features]

    model = xgb.XGBClassifier()
    model.fit(train_arr, train_target)
    proba = model.predict_proba(test_arr)
    classes = [x for x in model.classes_]
    for cl in classes:
        test[cl] = proba[:, classes.index(cl)]

    test[LISTING_ID] = test.index.values
    res = test[['listing_id', 'high', 'medium', 'low']]

    fp= 'stacking_blja_{}.csv'.format(int(time()))
    res.to_csv(fp, index=False)



def load_and_unite_submits_fs(experiments, folder=stacking_submit_fp):
    submit_names = ['submit_{}'.format(x)  for x in experiments]
    dfs = [(e, load_from_fs_avg_validation_df(e, folder)) for e in submit_names]
    targets = ['low', 'medium', 'high']
    res_df = None
    counter = 0
    for e, df in dfs:
        e=e.replace('submit_', '')
        if counter == 0:
            res_df = df[targets]
            res_df = res_df.rename(columns={k: '{}_{}'.format(e, k) for k in targets})
        else:
            tmp = df[targets]
            tmp = tmp.rename(columns={k: '{}_{}'.format(e, k) for k in targets})
            res_df = pd.merge(res_df, tmp, left_index=True, right_index=True)

        counter += 1

    return res_df

###################################################
#SUBMITTING....
###################################################
