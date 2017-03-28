import json
import os
from collections import OrderedDict
from math import log
from time import time

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_FAIL
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp, fmin
from hyperopt import tpe
from hyperopt.mongoexp import MongoTrials
from scipy.stats import boxcox
from sklearn.metrics import log_loss
from functools import partial
from sklearn.model_selection import StratifiedKFold
import math

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

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]

#========================================================
#MANAGER NUM


def process_manager_num(train_df, test_df):
    mngr_num_col = 'manager_num'
    df = train_df.groupby(MANAGER_ID)[MANAGER_ID].count()
    # df[df<=1]=-1
    df = df.apply(float)
    df = df.to_frame(mngr_num_col)
    train_df = pd.merge(train_df, df, left_on=MANAGER_ID, right_index=True)
    test_df = pd.merge(test_df, df, left_on=MANAGER_ID, right_index=True, how='left')

    return train_df, test_df, [mngr_num_col]

#========================================================

def hcc_encode(train_df, test_df, variable, binary_target, k=5, f=1, g=1, r_k = 0.01, folds=5):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    prior_prob = train_df[binary_target].mean()
    hcc_name = "_".join(["hcc", variable, binary_target])

    skf = StratifiedKFold(folds)
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        grouped = big.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
        grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
        grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

        if hcc_name in small.columns:
            del small[hcc_name]
        small = pd.merge(small, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
        small.loc[small[hcc_name].isnull(), hcc_name] = prior_prob
        small[hcc_name]= small[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(small))
        train_df.loc[small.index, hcc_name] = small[hcc_name]

    grouped = train_df.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    test_df = pd.merge(test_df, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
    test_df.loc[test_df[hcc_name].isnull(), hcc_name] = prior_prob

    return train_df, test_df, hcc_name


def get_exp_lambda(k,f):
    def res(n):
        return 1/(1+math.exp(float(k-n)/f))
    return res


def process_mngr_categ_preprocessing(train_df, test_df, k, f, n, r_k=0.01, g=1):
    col = MANAGER_ID
    new_cols=[]
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s=='high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s=='medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col, k=k, f=f, g=g, r_k=r_k, folds=n)
        new_cols.append(new_col)

    return train_df, test_df, new_cols




def with_lambda_loss(df, k, f, n):
    import json
    import os
    from collections import OrderedDict
    from math import log
    from time import time

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from hyperopt import STATUS_FAIL
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp, fmin
    from hyperopt import tpe
    from hyperopt.mongoexp import MongoTrials
    from scipy.stats import boxcox
    from sklearn.metrics import log_loss
    from functools import partial
    import math

    try:
        import dill as pickle
    except ImportError:
        import pickle

    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_month", "created_day", CREATED_HOUR, CREATED_MINUTE]

    train_df, test_df = split_df(df, 0.7)


    col = MANAGER_ID

    train_df, test_df, new_columns = process_manager_num(train_df, test_df)
    features+=new_columns

    train_df, test_df, new_columns = process_mngr_categ_preprocessing(train_df, test_df, k, f, n)
    features+=new_columns

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=1000, objective='mlogloss', subsample=0.8, colsample_bytree=0.8)
    # estimator = RandomForestClassifier(n_estimators=1000)
    estimator.fit(train_arr, train_target)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()


    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)
    loss = log_loss(test_target, proba)
    return loss




def loss_for_batch(s, df, runs):
    def log(ss):
        print ss


    t = time()

    f = s['f']
    k = s['k']
    n=int(s['n'])
    if k <= 1 or f <= 0.1:
        return {'loss': 1000, 'status': STATUS_FAIL}

    # print 'Running for k={}, f={}'.format(k,f)
    l = []
    for x in range(runs):
        loss = with_lambda_loss(df.copy(), k, f, n)
        print loss
        l.append(loss)

    t = time() - t


    avg_loss = np.mean(l)
    var = np.var(l)

    log([
        '\n\n',
        'summary for k={}, f={}, n={}'.format(k, f, n),
        'current_loss={}, best={}'.format(avg_loss, '?'),
        'time: {}'.format(t),
        'std={}'.format(np.std(l)),
        '\n\n'
    ])

    return {
        'loss': avg_loss,
        'loss_variance': var,
        'status': STATUS_OK,
        'losses_m': json.dumps(l),
        'params_m':json.dumps({'k': k, 'f': f, 'n':n}),
        'std_m':str(np.std(l))
    }