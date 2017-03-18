import json
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import boxcox
from sklearn.metrics import log_loss

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

FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)

# train_file = '../data/redhoop/train.json'
# test_file = '../data/redhoop/test.json'

train_file = '../../data/redhoop/train.json'
test_file = '../../data/redhoop/test.json'

import re

import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

BAD_SYMBOLS_REGEX= re.compile('[\.*-+&$#,\()\{\}\'~!?:;"\<\>]')
BAD_TOKEN_PATTERN = re.compile('.*\d.*')
BAD_TOKENS= {'\\', '-', '/'}

TOKENIZER = TweetTokenizer()

#TODO
#u'central a/c'


def lower_df(df):
    df[F_COL]=df[F_COL].apply(lambda l: [x.lower() for x in l])

def normalize_feature(s):
    s=s.lower()
    return BAD_SYMBOLS_REGEX.sub(' ', s)

def should_skip_token(s):
    if s in STOP_WORDS:
        return True

    if s in BAD_TOKENS:
        return True

    return re.match(BAD_TOKEN_PATTERN, s) is not None

def normalize_token_df(df):
    def l_to_tokens_list(l):
        s = ' '.join(l)
        s= normalize_feature(s)
        return TOKENIZER.tokenize(s)

    df[F_COL] = df[F_COL].apply(l_to_tokens_list)

    return df


def get_c_map_tokens(df, N=None):
    lower_df(df)
    tokenizer = TweetTokenizer()
    c_map={}
    for l in df[F_COL]:
        for f in l:
            f=normalize_feature(f)
            for t in tokenizer.tokenize(f):
                if should_skip_token(t):
                    continue
                if t in c_map:
                    c_map[t]+=1
                else:
                    c_map[t]=1


    c_map=[(k,v) for k,v in c_map.iteritems()]
    c_map.sort(key=lambda s:s[1], reverse=True)

    if N is None:
        N=len(c_map)

    return c_map[:N]

def get_top_N_counts_tokens(df, N=None):
    c_map_ordered = get_c_map_tokens(df)
    if N is None:
        N = len(c_map_ordered)
    return c_map_ordered[:N]

def get_top_N_tokens(df,N):
    return [x[0] for x in get_top_N_counts_tokens(df, N)]

def add_top_N_tokens_df(df, N):
    top_N = get_top_N_tokens(df, N)
    normalize_token_df(df)
    col_to_series={}
    new_cols=[]
    for f in top_N:
        s = df[F_COL].apply(lambda l: 1 if f in l else 0)
        new_col = val_to_col(f)
        col_to_series[new_col] = s
        new_cols.append(new_col)

    for col in df.columns.values:
        col_to_series[col] = df[col]

    return pd.DataFrame(col_to_series), new_cols

def add_tokens_list(df, l):
    normalize_token_df(df)
    for col in l:
        f = col_to_val(col)
        df[col] = df[F_COL].apply(lambda l: 1 if f in l else 0)

    return df

def val_to_col(s):
    return s.replace(' ', '_') + '_'

def col_to_val(col):
    return col.replace('_', ' ').strip()


def add_top_N_tokens_train_test(train_df, test_df, N):
    train_df, new_cols = add_top_N_tokens_df(train_df, N)
    test_df = add_tokens_list(test_df, new_cols)

    return train_df, test_df, new_cols


def out(l, loss, l_1K, loss1K, num, t):
    print '\n\n'
    print '#{}'.format(num)
    if loss1K is not None:
        print 'loss1K {}'.format(loss1K)
        print 'avg_loss1K {}'.format(np.mean(l_1K))
        print

    print 'loss {}'.format(loss)
    print 'avg_loss {}'.format(np.mean(l))
    print 'std {}'.format(np.std(l))
    print 'time {}'.format(t)

def write_results(l, fp):
    with open(fp, 'w+') as f:
        json.dump(l, f)


def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]


def load_train():
    return basic_preprocess(pd.read_json(train_file))


def load_test():
    return basic_preprocess(pd.read_json(test_file))

def process_outliers_lat_long(train_df, test_df):
    min_lat=40
    max_lat=41
    min_long=-74.1
    max_long=-73

    good_lat = (train_df[LATITUDE] < max_lat) & (train_df[LATITUDE] > min_lat)
    good_long = (train_df[LONGITUDE] < max_long) & (train_df[LONGITUDE] > min_long)

    train_df = train_df[good_lat & good_long]

    bed_lat = (test_df[LATITUDE] >=max_lat) | (test_df[LATITUDE] <=min_lat)
    bed_long = (test_df[LONGITUDE] >= max_long) | (test_df[LONGITUDE] <= min_long)
    test_df[LATITUDE][bed_lat] = train_df[LATITUDE].mean()
    test_df[LONGITUDE][bed_long]=train_df[LONGITUDE].mean()

    return train_df, test_df


def basic_preprocess(df):
    df['num_features'] = df[u'features'].apply(len)
    df['num_photos'] = df['photos'].apply(len)
    df['word_num_in_descr'] = df['description'].apply(lambda x: len(x.split(' ')))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    bc_price, tmp = boxcox(df['price'])
    df['bc_price'] = bc_price

    return df


# (0.61509489625789615, [0.61124170916042475, 0.61371758902339113, 0.61794752159334343, 0.61555861194203254, 0.61700904957028924])
def simple_loss(df):
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_month", "created_day"]

    train_df, test_df = split_df(df, 0.7)

    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=1000, objective='mlogloss')
    # estimator = RandomForestClassifier(n_estimators=1000)
    estimator.fit(train_arr, train_target)

    # plot feature importance
    # ffs= features[:len(features)-1]+['man_id_high', 'man_id_medium', 'man_id_low', 'manager_skill']
    # sns.barplot(ffs, [x for x in estimator.feature_importances_])
    # sns.plt.show()


    # print estimator.feature_importances_
    proba = estimator.predict_proba(test_arr)
    return log_loss(test_target, proba)

def get_loss_at1K(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    return results_on_test[1000]

def loss_with_per_tree_stats(df, num):
    features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                'num_features', 'num_photos', 'word_num_in_descr',
                "created_year", "created_month", "created_day"]

    train_df, test_df = split_df(df, 0.7)
    train_df, test_df, new_cols = add_top_N_tokens_train_test(train_df, test_df, num)
    features+=new_cols
    print new_cols


    train_target, test_target = train_df[TARGET].values, test_df[TARGET].values
    del train_df[TARGET]
    del test_df[TARGET]

    train_df = train_df[features]
    test_df = test_df[features]

    train_arr, test_arr = train_df.values, test_df.values

    estimator = xgb.XGBClassifier(n_estimators=1500, objective='mlogloss')
    # estimator = RandomForestClassifier(n_estimators=1000)
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
    return loss, loss1K, xgboost_per_tree_results(estimator)

def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['mlogloss']
    results_on_train = estimator.evals_result()['validation_0']['mlogloss']
    return {
        'train':results_on_train,
        'test':results_on_test
    }


def do_test(num, fp):
    l = []
    train_df = load_train()
    for x in range(num):
        t=time()
        df=train_df.copy()

        loss = simple_loss(df)
        t=time()-t
        l.append(loss)

        out(l, loss,None,None, x, t)
        write_results(l, fp)

def do_test_with_xgboost_stats_per_tree(num, fp):
    l = []
    results =[]
    l_1K=[]
    train_df = load_train()
    for x in range(num):
        t=time()
        df=train_df.copy()

        loss, loss1K, res = loss_with_per_tree_stats(df, 100)
        t=time()-t
        l.append(loss)
        l_1K.append(loss1K)
        results.append(res)

        out(l, loss, l_1K, loss1K, x, t)
        write_results(results, fp)


# train_df, test_df = load_train(), load_test()

do_test_with_xgboost_stats_per_tree(1000, 'top_100_tokens_naive.json')
