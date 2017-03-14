import json
import os
import sys
from functools import partial
from math import log
from time import time

import numpy as np
import pandas as pd
from hyperopt import hp, fmin
from hyperopt import tpe
from hyperopt.mongoexp import MongoTrials
from scipy.stats import boxcox

from bid_and_mngr_optimizer import with_lambda_loss

train_file = '../../data/redhoop/train.json'
test_file = '../../data/redhoop/test.json'


def split_df(df, c):
    msk = np.random.rand(len(df)) < c
    return df[msk], df[~msk]


def load_train():
    return basic_preprocess(pd.read_json(train_file))


def load_test():
    return basic_preprocess(pd.read_json(test_file))

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

def write_losses_test(num, fp):
    df = load_train()
    l=[]
    for i in range(num):
        t=time()
        loss = with_lambda_loss(df.copy())
        l.append(loss)
        with open(fp, 'w+') as fl:
            json.dump(l, fl)
        print '\n\n'
        print '#{}'.format(i)
        print 'loss = {}'.format(loss)
        print 'avg  = {}'.format(np.mean(l))
        print 'std  = {}'.format(np.std(l))
        print 'time = {}'.format(time()-t)


write_losses_test(500, '/home/dpetrovskyi/PycharmProjects/kaggle/trash/bid_and_mngr/bid_and_mngr.json')