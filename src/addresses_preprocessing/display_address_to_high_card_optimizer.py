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
BID_ZERO='bid_zero'

DISPLAY_ADDRESS_NORMALIZED = 'display_address_normalized'


FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

NORMALIZATION_MAP = {
    'street': ['St', 'St.', 'Street', 'St,'],
    'avenue': ['Avenue', 'Ave', 'Ave.'],
    'square': ['Square'],
    'east': ['e', 'east'],
    'west': ['w', 'west']
}


def reverse_norm_map(m):
    res = {}
    for k, v in m.iteritems():
        for s in v:
            res[s.lower()] = k.lower()

    return res


REVERSE_NORM_MAP = reverse_norm_map(NORMALIZATION_MAP)

def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]


def process_with_lambda(train_df, test_df, col, target_col, target_vals, lambda_f):
    temp_target = '{}_'.format(target_col)
    train_df[temp_target]= train_df[target_col]
    train_df= pd.get_dummies(train_df, columns=[target_col])
    dummies_cols = [dummy_col(target_col, v) for v in target_vals]
    priors = train_df[dummies_cols].mean()
    priors_arr = [priors[dummy_col(target_col, v)] for v in target_vals]
    agg = OrderedDict(
        [(dummy_col(target_col, v), OrderedDict([('{}_mean'.format(v),'mean')])) for v in target_vals] + [(col, {'cnt':'count'})]
    )
    df = train_df[[col]+dummies_cols].groupby(col).agg(agg)
    df.columns = ['posterior_{}'.format(v) for v in target_vals] + ['cnt']
    new_cols=[]
    for v in target_vals:
        def norm_posterior(x):
            cnt= float(x['cnt'])
            posterior = x['posterior_{}'.format(v)]
            prior = priors[dummy_col(target_col, v)]
            l = lambda_f(cnt)
            return (l * posterior) + ((1 - l) * prior)

        new_col = '{}_coverted_exp_for_{}={}'.format(col, target_col, v)
        df[new_col] =df.apply(norm_posterior, axis=1)
        new_cols.append(new_col)

    df = df[new_cols]

    train_df = pd.merge(train_df, df, left_on=col, right_index=True)

    test_df = pd.merge(test_df, df, left_on=col, right_index=True, how='left')
    test_df.loc[test_df[new_cols[0]].isnull(), new_cols] = priors_arr

    for c in dummies_cols:
        del train_df[c]

    train_df[target_col]= train_df[temp_target]
    del train_df[temp_target]

    return train_df, test_df

def cols(col, target_col, target_vals):
    return ['{}_coverted_exp_for_{}={}'.format(col, target_col, v) for v in target_vals]

def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)

def normalize_tokens(s):
    tokens = s.split()
    for i in range(len(tokens)):
        tokens[i] = if_starts_with_digit_return_digit_prefix(tokens[i])
        t = tokens[i]
        if t.lower() in REVERSE_NORM_MAP:
            tokens[i] = REVERSE_NORM_MAP[t.lower()]
    return ' '.join(tokens)

def if_starts_with_digit_return_digit_prefix(s):
    if not s[0].isdigit():
        return s
    last=0
    for i in range(len(s)):
        if s[i].isdigit():
            last=i
        else:
            break

    return s[0:last+1]


def normalize(s):
    s = normalize_tokens(s)
    if s == '':
        return s

    s=s.lower()

    tokens = s.split()
    if len(tokens) == 2:
        return ' '.join(tokens)
    if tokens[0].replace('.', '').replace('-', '').isdigit():
        return ' '.join(tokens[1:])
    else:
        return ' '.join(tokens)

def normalize_display_address(train_df, test_df):
    train_df[DISPLAY_ADDRESS_NORMALIZED] = train_df[DISPLAY_ADDRESS].apply(normalize)
    test_df[DISPLAY_ADDRESS_NORMALIZED] = test_df[DISPLAY_ADDRESS].apply(normalize)

    return train_df, test_df


def with_lambda_loss(df, k, f):
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
        # print('Went with dill')
    except ImportError:
        print 'Picckle blja'
        import pickle

    col = DISPLAY_ADDRESS_NORMALIZED
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"] + \
               cols(col, TARGET, TARGET_VALUES)

    train_df, test_df = split_df(df, 0.7)
    train_df, test_df = normalize_display_address(train_df, test_df)
    lamdba_f = get_exp_lambda(k, f)
    train_df, test_df = process_with_lambda(train_df, test_df, col, TARGET, TARGET_VALUES, lamdba_f)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=1000, objective='mlogloss')
    estimator.fit(train_arr, train_target)

    proba = estimator.predict_proba(test_arr)
    loss = log_loss(test_target, proba)
    return loss

def get_exp_lambda(k,f):
    def res(n):
        return 1/(1+math.exp(float(k-n)/f))
    return res


def loss_for_batch(s, df=None, runs=None):
    def log(ss):
        print ss


    t = time()

    f = s['f']
    k = s['k']
    if k <= 1 or f <= 0.1:
        return {'loss': 1000, 'status': STATUS_FAIL}

    # print 'Running for k={}, f={}'.format(k,f)
    l = []
    for x in range(runs):
        loss = with_lambda_loss(df.copy(), k, f)
        print loss
        l.append(loss)

    t = time() - t


    avg_loss = np.mean(l)
    var = np.var(l)

    log([
        '\n\n',
        'summary for k={}, f={}'.format(k, f),
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
        'params_m':json.dumps({'k': k, 'f': f}),
        'std_m':str(np.std(l))
    }