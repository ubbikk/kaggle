import json
import os
import seaborn as sns
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.metrics import log_loss
import numpy as np
import xgboost as xgb

sns.set(color_codes=True)
os.chdir('/home/dpetrovskyi/PycharmProjects/kaggle/src')

train_file = '../data/redhoop/train.json'
test_file = '../data/redhoop/test.json'


def perform_cross_val_with_xgboost(data, target):
    model = xgb.XGBClassifier()
    kf = KFold(len(target), n_folds=5)
    res = []
    for train_ind, test_ind in kf:
        data_train = data[train_ind]
        data_test = data[test_ind]

        target_train = target[train_ind]
        target_test = target[test_ind]

        model.fit(data_train, target_train)
        proba = model.predict_proba(data_test)
        l = log_loss(target_test, proba)
        res.append(l)

    return np.mean(res), res


def perform_cross_val(estimator, data, target):
    kf = KFold(len(target), n_folds=5)
    res = []
    for train_ind, test_ind in kf:
        data_train = data[train_ind]
        data_test = data[test_ind]

        target_train = target[train_ind]
        target_test = target[test_ind]

        estimator.fit(data_train, target_train)
        proba = estimator.predict_proba(data_test)
        l = log_loss(target_test, proba)
        res.append(l)

    return np.mean(res), res

def score_for_const(obj):
    t_dist = explore_target(obj)
    t_dist = {k: float(v)/len(obj) for k,v in t_dist.iteritems()}
    keys = t_dist.keys()
    probs = [t_dist[k] for k in keys]

    target = np.array([v['interest_level'] for v in obj.values()]).reshape(len(obj),)
    kf = KFold(len(target), n_folds=5)
    res = []
    for train_ind, test_ind in kf:
        target_test = target[test_ind]
        proba = np.array([probs for x in range(len(target_test))]).reshape(len(target_test), len(keys))
        def blja(s):
            return [1 if k==s else 0 for k in keys]

        y_true = np.array([blja(s) for s in target_test]).reshape(len(target_test), len(keys))

        l = log_loss(y_true, proba)
        res.append(l)

    return np.mean(res), res

def load_train():
    with open(train_file) as f:
        return convert(json.load(f))


def load_test():
    with open(test_file) as f:
        return convert(json.load(f))


def convert(obj):
    res = {}
    for field, values in obj.iteritems():
        for k, v in values.iteritems():
            if k not in res:
                res[k] = {}

            res[k][field] = v

    return res


def all_interest_levels(obj):
    res = set()
    for v in obj.values():
        res.add(v[u'interest_level'])
    return res


def all_features(obj):
    res = set()
    for v in obj.values():
        res.update(v['features'])
    return res


def sorted_features(obj):
    res = {k: 0 for k in all_features(obj)}
    for v in obj.values():
        for f in v['features']:
            res[f] += 1
    res = [(f, num) for f, num in res.iteritems()]
    res.sort(key=lambda s: s[1], reverse=True)

    return res


def features_with_at_least_occurences(obj, min_occurences):
    ff = sorted_features(obj)
    ff = filter(lambda s: s[1] <= min_occurences, ff)
    return [x[0] for x in ff]


def all_fields(name, obj):
    res = set()
    for v in obj.values():
        res.add(v[name])
    return res


def prices(obj):
    return [x['price'] for x in obj.values()]


def prices_hisr(obj, cutoff=10000):
    p = prices(obj)
    p = filter(lambda s: s < cutoff, p)
    sns.distplot(p, bins=100, kde=False)


def too_big_prices(obj, cutoff):
    p = prices(obj)
    return filter(lambda s: s > cutoff, p)


def too_small_prices(obj, cutoff):
    p = prices(obj)
    return filter(lambda s: s < cutoff, p)


# def convert_to_features_naive(obj):
#     features_to_copy = [u'longitude',u'latitude' ,u'bedrooms' ,u'bathrooms', 'price', u'interest_level']
#     ff =  features_with_at_least_occurences(obj, 1000)
#     res = {k:{} for k in obj.keys()}
#
#     for k,v in obj.iteritems():
#         for s in features_to_copy:
#             res[k][s] = v[s]
#
#         for f in ff:
#             res[k][f] = 0
#
#         for f in v['features']:
#             res[k][f] = 1
#
#     return res

# def add_some_additional_fields(obj):


def convert_to_features_with_additional_fields(obj):
    target_name = 'interest_level'
    for k,v in obj.iteritems():
        v['photo_num'] = len(v['photos'])
    features_to_copy = [u'longitude', u'latitude', u'bedrooms', u'bathrooms', 'price', 'photo_num']
    features_to_col = {features_to_copy[i]: i for i in range(len(features_to_copy))}
    sz = len(obj)
    target = [0 for x in range(sz)]
    data = [[0 for y in range(len(features_to_copy))] for x in range(sz)]

    counter = -1
    for k,v in obj.iteritems():
        counter+=1
        target[counter] = v[target_name]
        for s in features_to_copy:
            ind = features_to_col[s]
            data[counter][ind] = v[s]

    return np.array(data), np.array(target).reshape(sz,)


def convert_to_features_very_naive(obj):
    target_name = 'interest_level'
    for k,v in obj.iteritems():
        v['photo_num'] = len(v['photos'])
    features_to_copy = [u'longitude', u'latitude', u'bedrooms', u'bathrooms', 'price', 'photo_num']
    features_to_col = {features_to_copy[i]: i for i in range(len(features_to_copy))}
    sz = len(obj)
    target = [0 for x in range(sz)]
    data = [[0 for y in range(len(features_to_copy))] for x in range(sz)]

    counter = -1
    for k,v in obj.iteritems():
        counter+=1
        target[counter] = v[target_name]
        for s in features_to_copy:
            ind = features_to_col[s]
            data[counter][ind] = v[s]

    return np.array(data), np.array(target).reshape(sz,)

def convert_to_features_naive_csr(obj):
    target_name = 'interest_level'
    target = []
    for k,v in obj.iteritems():
        v['photo_num'] = len(v['photos'])
    features_to_copy = [u'longitude', u'latitude', u'bedrooms', u'bathrooms', 'price', 'photo_num']
    ff = features_with_at_least_occurences(obj, 1000)
    feature_to_col = {ff[i]: len(features_to_copy) + i for i in range(len(ff))}
    for i in range(len(features_to_copy)):
        feature_to_col[features_to_copy[i]] = i

    rows = []
    cols = []
    data = []
    counter = -1
    for k, v in obj.iteritems():
        counter += 1
        target += [v[target_name]]
        for f in features_to_copy:
            rows += [counter]
            cols += [feature_to_col[f]]
            data += [v[f]]

        for f in v['features']:
            if f not in ff:
                continue

            rows += [counter]
            cols += [feature_to_col[f]]
            data += [1]

    rows_cols = np.array([rows, cols]).reshape(2, len(rows))

    return csr_matrix((data, rows_cols)), np.array(target).reshape(len(target), )


def explore_target(obj):
    res = {}
    for v in obj.values():
        t = v['interest_level']
        if t in res:
            res[t] += 1
        else:
            res[t] = 1

    return res

# perform_cross_val(xgb.XGBClassifier(n_estimators=300), *convert_to_features_naive_csr(load_train()))
