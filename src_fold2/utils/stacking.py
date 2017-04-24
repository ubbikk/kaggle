import json
import math
from time import time, ctime

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

#100\1\1
# [('avg', 0.50343355521592148),
#  ('losses',
#   [0.50300851757426812,
#    0.50854799455858868,
#    0.50179941660459038,
#    0.49843412249451141,
#    0.50537772484764865])]


#100\0.8\0.8
# [('avg', 0.50314508236119893),
#  ('losses',
#   [0.50254642769754998,
#    0.50803737832627593,
#    0.50175941851710271,
#    0.49786438547185091,
#    0.50551780179321515])]


#100/1/1 +rnd_forest
# [('avg', 0.50268986529085102),
#  ('losses',
#   [0.50199733742705421,
#    0.50740274435941213,
#    0.50101593232235364,
#    0.49787455105051448,
#    0.50515876129492054])]


#100/0.8/0.8
# [('avg', 0.50263676242897237),
#  ('losses',
#   [0.50225325920570485,
#    0.50768087578571286,
#    0.50071042068922744,
#    0.49709903467159272,
#    0.50544022179262393])]


#100/0.8/0.8
# [('avg', 0.50114397549622214),
#  ('losses',
#   [0.50040989328030627,
#    0.50631434801408015,
#    0.49921892655985017,
#    0.49626061278835965,
#    0.50351609683851462])]




def run_xgb_cv(experiments, train_df, folder=stacking_fp):
    data = create_data_for_running_fs(experiments, train_df, folder)
    losses = []
    run_results = []
    imp =[]
    features = None
    for train, test, train_target, test_target in data:
        # train, test = process_avg_mngr_score(train, test, train_df)
        eval_set = [(train.values, train_target), (test.values, test_target)]
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=150,
            colsample_bytree=0.8,
            subsample=0.8,
            seed=int(time())
        )
        model.fit(train.values, train_target, eval_set=eval_set, eval_metric='mlogloss', verbose=False)
        proba = model.predict_proba(test.values)
        loss = log_loss(test_target, proba)
        print loss
        losses.append(loss)
        run_results.append(xgboost_per_tree_results(model))
        imp.append(model.feature_importances_)

    plot_errors_xgboost(run_results)
    # plot_importance(imp, features)
    return [
        ('avg', np.mean(losses)),
        ('losses', losses)
    ]


def process_avg_mngr_score(train, test, train_df):
    MANAGER_ID = 'manager_id'
    df = pd.concat([train, test])

    df = pd.merge(df, train_df[[MANAGER_ID]], left_index=True, right_index=True)
    cols = ['stacking_all_{}'.format(x) for x in ['low', 'medium', 'high']]
    new_cols = []
    for col in cols:
        new_col = 'mngr_{}'.format(col)
        df[new_col] = df.groupby(MANAGER_ID)[col].mean()
        new_cols.append(new_col)

    del df[MANAGER_ID]


    return df.loc[train.index], df.loc[test.index]


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


def plot_importance(arr, features, N=None):
    if N is None:
        N=len(features)

    sz = len(features)
    res = [np.mean([x[j] for x in arr]) for j in range(sz)]
    stds = [np.std([x[j] for x in arr]) for j in range(sz)]
    res = zip(features, res, stds)
    res.sort(key=lambda s: s[1], reverse=True)
    print res
    res=res[:N]
    xs = [x[0] for x in res]
    ys=[x[1] for x in res]
    sns.barplot(xs, ys)
    sns.plt.show()


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
            res = pd.read_csv(os.path.join(folder, f), index_col=LISTING_ID)
            print name, len(res)
            return res

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

def create_data_for_running_fs_with_mngr(experiments, train_df, folder=stacking_fp):
    MANAGER_ID = 'manager_id'
    df = load_and_unite_expiriments_fs(experiments, folder)
    res = []
    for cv in range(CV):
        small_indexes = SPLITS_SMALL[cv]
        big_indexes = SPLITS_BIG[cv]
        train = df.loc[big_indexes]
        train_target = train_df.loc[big_indexes][TARGET]
        train[MANAGER_ID] = train_df.loc[big_indexes][MANAGER_ID]
        test_target = train_df.loc[small_indexes][TARGET]
        test = df.loc[small_indexes]
        test[MANAGER_ID] = train_df.loc[small_indexes][MANAGER_ID]
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

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        colsample_bytree=0.8,
        subsample=0.8,
        seed=int(time())
    )
    model.fit(train_arr, train_target)
    proba = model.predict_proba(test_arr)
    classes = [x for x in model.classes_]
    for cl in classes:
        test[cl] = proba[:, classes.index(cl)]

    test[LISTING_ID] = test.index.values
    res = test[['listing_id', 'high', 'medium', 'low']]

    fp= 'stacking__{}.csv'.format(time_now_str())
    res.to_csv(fp, index=False)



def load_and_unite_submits_fs(experiments, folder=stacking_submit_fp):
    submit_names = ['sub_{}'.format(x)  for x in experiments]
    dfs = [(e, load_from_fs_avg_validation_df(e, folder)) for e in submit_names]
    targets = ['low', 'medium', 'high']
    res_df = None
    counter = 0
    for e, df in dfs:
        e=e.replace('sub_', '')
        if counter == 0:
            res_df = df[targets]
            res_df = res_df.rename(columns={k: '{}_{}'.format(e, k) for k in targets})
        else:
            tmp = df[targets]
            tmp = tmp.rename(columns={k: '{}_{}'.format(e, k) for k in targets})
            res_df = pd.merge(res_df, tmp, left_index=True, right_index=True)

        counter += 1

    return res_df

def time_now_str():
    return str(ctime()).replace(' ', '_')

###################################################
#SUBMITTING....
###################################################
