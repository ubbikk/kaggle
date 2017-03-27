import json
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import product
from sklearn.model_selection import StratifiedKFold


#my



TARGET = u'interest_level'
TARGET_VALUES = ['low', 'medium', 'high']
MANAGER_ID = 'manager_id'
BUILDING_ID = 'building_id'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
PRICE = 'price'
BATHROOMS = 'bathrooms'
BEDROOMS = 'bedrooms'
DESCRIPTION = 'description'
DISPLAY_ADDRESS = 'display_address'
STREET_ADDRESS = 'street_address'
LISTING_ID = 'listing_id'
PRICE_PER_BEDROOM = 'price_per_bedroom'
F_COL=u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'

# train_file = '../data/redhoop/train.json'
# test_file = '../data/redhoop/test.json'

train_file = '../../data/redhoop/train.json'
test_file = '../../data/redhoop/test.json'


def basic_preprocess(df):
    df['num_features'] = df[u'features'].apply(len)
    df['num_photos'] = df['photos'].apply(len)
    df['word_num_in_descr'] = df['description'].apply(lambda x: len(x.split(' ')))
    df["created"] = pd.to_datetime(df["created"])
    # df["created_year"] = df["created"].dt.year
    df[CREATED_MONTH] = df["created"].dt.month
    df[CREATED_DAY] = df["created"].dt.day
    df[CREATED_HOUR] = df["created"].dt.hour
    df[CREATED_MINUTE] = df["created"].dt.minute
    df[DAY_OF_WEEK] = df['created'].dt.dayofweek
    # bc_price, tmp = boxcox(df['price'])
    # df['bc_price'] = bc_price

    return df

def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]


def load_train():
    return basic_preprocess(pd.read_json(train_file))


def load_test():
    return basic_preprocess(pd.read_json(test_file))



#my


def add_features(df):
    fmt = lambda s: s.replace("\u00a0", "").strip().lower()
    df["photo_count"] = df["photos"].apply(len)
    df["street_address"] = df['street_address'].apply(fmt)
    df["display_address"] = df["display_address"].apply(fmt)
    df["desc_wordcount"] = df["description"].apply(len)
    df["pricePerBed"] = df['price'] / df['bedrooms']
    df["pricePerBath"] = df['price'] / df['bathrooms']
    df["pricePerRoom"] = df['price'] / (df['bedrooms'] + df['bathrooms'])
    df["bedPerBath"] = df['bedrooms'] / df['bathrooms']
    df["bedBathDiff"] = df['bedrooms'] - df['bathrooms']
    df["bedBathSum"] = df["bedrooms"] + df['bathrooms']
    df["bedsPerc"] = df["bedrooms"] / (df['bedrooms'] + df['bathrooms'])

    df = df.fillna(-1).replace(np.inf, -1)
    return df


def factorize(df1, df2, column):
    ps = df1[column].append(df2[column])
    factors = ps.factorize()[0]
    df1[column] = factors[:len(df1)]
    df2[column] = factors[len(df1):]
    return df1, df2


def designate_single_observations(df1, df2, column):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df1.loc[df1.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    df2.loc[df2.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    return df1, df2


def hcc_encode(train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    hcc_name = "_".join(["hcc", variable, target])

    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    train_df = pd.merge(train_df, grouped[[hcc_name]], left_on=variable, right_index=True)
    train_df[hcc_name]= train_df[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(train_df))

    test_df = pd.merge(test_df, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
    test_df.loc[test_df[hcc_name].isnull(), hcc_name] = prior_prob
    # test_df[hcc_name]= test_df[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(test_df))

    return train_df, test_df

    # df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
    # if r_k: df *= np.random.uniform(1 - r_k, 1 + r_k, len(test_df))     # Add uniform noise. Not mentioned in original paper
    #
    # if update_df is None: update_df = test_df
    # if hcc_name not in update_df.columns: update_df[hcc_name] = np.nan
    # update_df.update(df)


# Load data
# X_train = pd.read_json("../data/redhoop/train.json").sort_values(by="listing_id")
# X_test = pd.read_json("../data/redhoop/test.json").sort_values(by="listing_id")

# X_train = pd.read_json("../../data/redhoop/train.json").sort_values(by="listing_id")
# X_test = pd.read_json("../../data/redhoop/test.json").sort_values(by="listing_id")

# Make target integer, one hot encoded, calculate target priors

def do_is_it_lit(X_train, X_test):
    X_train = X_train.replace({"interest_level": {"low": 0, "medium": 1, "high": 2}})
    X_train = X_train.join(pd.get_dummies(X_train["interest_level"], prefix="pred").astype(int))
    prior_0, prior_1, prior_2 = X_train[["pred_0", "pred_1", "pred_2"]].mean()

    # Add common features
    X_train = add_features(X_train)
    X_test = add_features(X_test)

    # Special designation for building_ids, manager_ids, display_address with only 1 observation
    for col in ('building_id', 'manager_id', 'display_address'):
        X_train, X_test = designate_single_observations(X_train, X_test, col)

    # High-Cardinality Categorical encoding
    attributes = product(("building_id", "manager_id"), zip(("pred_1", "pred_2"), (prior_1, prior_2)))
    for variable, (target, prior) in attributes:
        X_train, X_test= hcc_encode(X_train, X_test, variable, target, prior, k=5, r_k=0.01)
        # for train, test in skf.split(np.zeros(len(X_train)), X_train['interest_level']):
        #     hcc_encode(X_train.iloc[train], X_train.iloc[test], variable, target, prior, k=5, r_k=0.01, update_df=X_train)

    # Factorize building_id, display_address, manager_id, street_address
    for col in ('building_id', 'display_address', 'manager_id', 'street_address'):
        X_train, X_test = factorize(X_train, X_test, col)

    # Create binarized features
    fmt = lambda feat: [s.replace("\u00a0", "").strip().lower().replace(" ", "_") for s in feat]  # format features
    X_train["features"] = X_train["features"].apply(fmt)
    X_test["features"] = X_test["features"].apply(fmt)
    features = [f for f_list in list(X_train["features"]) + list(X_test["features"]) for f in f_list]
    ps = pd.Series(features)
    grouped = ps.groupby(ps).agg(len)
    features = grouped[grouped >= 10].index.sort_values().values    # limit to features with >=10 observations
    mlb = MultiLabelBinarizer().fit([features])
    columns = ['feature_' + s for s in mlb.classes_]
    flt = lambda l: [i for i in l if i in mlb.classes_]     # filter out features not present in MultiLabelBinarizer
    X_train = X_train.join(pd.DataFrame(data=mlb.transform(X_train["features"].apply(flt)), columns=columns, index=X_train.index))
    X_test = X_test.join(pd.DataFrame(data=mlb.transform(X_test["features"].apply(flt)), columns=columns, index=X_test.index))

    X_test = X_test.replace({"interest_level": {"low": 0, "medium": 1, "high": 2}})
    return X_train, X_test





# Save
"""
X_train = X_train.sort_index(axis=1).sort_values(by="listing_id")
X_test = X_test.sort_index(axis=1).sort_values(by="listing_id")
columns_to_drop = ["photos", "pred_0","pred_1", "pred_2", "description", "features", "created"]
X_train.drop([c for c in X_train.columns if c in columns_to_drop], axis=1).\
    to_csv("data/train_python.csv", index=False, encoding='utf-8')
X_test.drop([c for c in X_test.columns if c in columns_to_drop], axis=1).\
    to_csv("data/test_python.csv", index=False, encoding='utf-8')
"""
import xgboost as xgb
import math

columns_to_drop = ["photos", "pred_0","pred_1", "pred_2", "description", "features", "created"]
TARGET = u'interest_level'
TARGET_VALUES = ['low', 'medium', 'high']
MANAGER_ID = 'manager_id'
BUILDING_ID = 'building_id'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
PRICE = 'price'
BATHROOMS = 'bathrooms'
BEDROOMS = 'bedrooms'
DESCRIPTION = 'description'
DISPLAY_ADDRESS = 'display_address'
STREET_ADDRESS = 'street_address'
LISTING_ID = 'listing_id'
PRICE_PER_BEDROOM = 'price_per_bedroom'
F_COL=u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'

def out(l, loss, l_1K, loss1K, num, t):
    print '\n\n'
    print '#{}'.format(num)
    if loss1K is not None:
        print 'loss1K {}'.format(loss1K)
        print 'avg_loss1K {}'.format(np.mean(l_1K))
        print get_3s_confidence_for_mean(l_1K)
        print

    print 'loss {}'.format(loss)
    print 'avg_loss {}'.format(np.mean(l))
    print get_3s_confidence_for_mean(l)
    print 'std {}'.format(np.std(l))
    print 'time {}'.format(t)

def get_3s_confidence_for_mean(l):
    std = np.std(l)/math.sqrt(len(l))
    m = np.mean(l)
    start = m -3*std
    end = m+3*std

    return '3s_confidence: [{}, {}]'.format(start, end)

def write_results(l, fp):
    with open(fp, 'w+') as f:
        json.dump(l, f)



def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]

def get_loss_at1K(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    return results_on_test[1000]

def loss_with_per_tree_stats(df):
    train_df, test_df = split_df(df, 0.7)
    train_df, test_df = do_is_it_lit(train_df, test_df)
    for df in [train_df, test_df]:
        for col in columns_to_drop:
            if col in df.columns.values:
                del df[col]

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    features = train_df.columns.values

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values
    print features

    estimator = xgb.XGBClassifier(n_estimators=1500, objective='mlogloss')
    eval_set = [(train_arr, train_target), (test_arr, test_target)]
    estimator.fit(train_arr, train_target, eval_set=eval_set, eval_metric='mlogloss', verbose=False)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()


    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)

    loss = log_loss(test_target, proba)
    loss1K = get_loss_at1K(estimator)
    return loss, loss1K, xgboost_per_tree_results(estimator), estimator.feature_importances_

def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    results_on_train = estimator.evals_result()['validation_0']['mlogloss']
    return {
        'train':results_on_train,
        'test':results_on_test
    }

def do_test_with_xgboost_stats_per_tree(num, fp):

    l = []
    results =[]
    l_1K=[]
    ii=[]
    train_df = load_train()

    for x in range(num):
        t=time()
        df=train_df.copy()

        loss, loss1K, res , imp= loss_with_per_tree_stats(df)
        ii.append(imp.tolist())

        t=time()-t
        l.append(loss)
        l_1K.append(loss1K)
        results.append(res)

        out(l, loss, l_1K, loss1K, x, t)
        write_results(results, fp)
        write_results(ii, fp+'_importance.json')


do_test_with_xgboost_stats_per_tree(1000, 'is_it_lit_drop_mgr_bid_etc.json')

