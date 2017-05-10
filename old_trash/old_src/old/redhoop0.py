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
import pandas as pd

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


def transform_to_csr(obj, ff, with_target=True, col_num=None):
    target_name = 'interest_level'
    target = []
    for k,v in obj.iteritems():
        v['photo_num'] = len(v['photos'])
    features_to_copy = [u'longitude', u'latitude', u'bedrooms', u'bathrooms', 'price', 'photo_num']
    feature_to_col = {ff[i]: len(features_to_copy) + i for i in range(len(ff))}
    for i in range(len(features_to_copy)):
        feature_to_col[features_to_copy[i]] = i

    if col_num is None:
        col_num = len(feature_to_col)

    rows = []
    cols = []
    data = []
    counter = -1
    for k, v in obj.iteritems():
        counter += 1
        if with_target:
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
    if with_target:
        return csr_matrix((data, rows_cols), shape=(len(obj), col_num)), np.array(target).reshape(len(target), )
    else:
        return csr_matrix((data, rows_cols), shape=(len(obj), col_num))

def convert_to_features_naive_csr(obj):
    ff = features_with_at_least_occurences(obj, 1000)
    return transform_to_csr(obj, ff, with_target=True)


def perform_xgboost():
    train = load_train()
    test = load_test()
    ff = features_with_at_least_occurences(train, 1000)
    data_train, target_train = transform_to_csr(train, ff, with_target=True, col_num=6+len(ff))
    data_test = transform_to_csr(test, ff, with_target=False, col_num=6+len(ff))
    estimator = xgb.XGBClassifier(n_estimators=1000)
    estimator.fit(data_train, target_train)
    probs = estimator.predict_proba(data_test)
    classes = estimator.classes_

    list_id = [v['listing_id'] for k,v in test.iteritems()]
    # arr = np.hstack((np.array(list_id).reshape(len(test), 1), probs))
    df = pd.DataFrame(data=probs, index=list_id, columns=[]+[x for x in classes])
    df.index.name = 'listing_id'
    df.to_csv('/home/dpetrovskyi/PycharmProjects/kaggle/src/results.csv', index=True)



def explore_target(obj):
    res = {}
    for v in obj.values():
        t = v['interest_level']
        if t in res:
            res[t] += 1
        else:
            res[t] = 1

    return res

def explore_building_id(obj):
    res={}
    for v in obj.values():
        b_id = v[u'building_id']
        if b_id not in res:
            res[b_id]=1
        else:
            res[b_id]+=1

    return res

def common_in_dicts(a, b):
    k_counter=0
    a_occurences_counter=0
    b_occurences_counter=0
    for k in a.keys():
        if k in b:
            k_counter+=1
            a_occurences_counter+=a[k]
            b_occurences_counter+=b[k]

    return k_counter, a_occurences_counter, b_occurences_counter

def build_id_len_distribution(obj):
    d = explore_building_id(obj)
    num_to_num_of_id = {}
    for b_id, num in d.iteritems():
        if num not in num_to_num_of_id:
            num_to_num_of_id[num] = 1
        else:
            num_to_num_of_id[num] +=1

    b = [[k,v] for k,v in num_to_num_of_id.iteritems()]
    b = pd.DataFrame(data=b, columns=['mentions', 'buildings'])
    # b.sort(key=lambda s:s[0])
    sns.jointplot(x="mentions", y="buildings", data=b);



# perform_cross_val(xgb.XGBClassifier(n_estimators=300), *convert_to_features_naive_csr(load_train()))
# perform_xgboost()
# build_id_len_distribution(load_train())
print score_for_const(load_train())