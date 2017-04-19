import json

import numpy as np
from numpy import mean, std
from sklearn.metrics import log_loss
from scipy.stats import normaltest
import math
from time import time
import seaborn as sns

from sklearn.model_selection import StratifiedKFold

TARGET = 'interest_level'

def split_df(df, seed):
    folds=5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    gen = skf.split(np.zeros(len(df)), df[TARGET])
    v = np.random.randint(5)
    counter=0
    for big_ind, small_ind in gen:
        if counter == v:
            return df.iloc[big_ind], df.iloc[small_ind]
        counter+=1

    raise

def get_probs_df(train_df, probs):
    probs[TARGET] = train_df[TARGET]
    return probs

def explore_res(arr):
    s = std(arr) / math.sqrt(len(arr))
    m = mean(arr)
    return [
        ('avg_loss  ', m),
        ('max       ', max(arr)),
        ('min       ', min(arr)),
        ('2s        ', '[{}, {}]'.format(m-2*s, m+2*s)),
        ('3s        ', '[{}, {}]'.format(m-3*s, m+3*s)),
        ('norm      ', normaltest(arr).pvalue),
        ('std       ', np.std(arr)),
        ('mean_std  ', s),
        ('len       ', len(arr))
    ]


def get_log_losses(train_df, probs, N=1000):
    df = get_probs_df(train_df, probs)
    losses=[]
    t = int(time())
    for x in range(N):
        big, small = split_df(df, t+x)
        l = log_loss(small[TARGET], small[['high', 'low', 'medium']])
        losses.append(l)

    return losses

def visualize_res(res):
    sns.distplot(res)

