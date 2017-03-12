import pandas as pd
from collections import OrderedDict
import math


def cols_forM(col, target_col, target_vals, m):
    m = float(m)
    return ['{}_catM={}_for_{}={}'.format(col, m, target_col, v) for v in target_vals]

def cols(col, target_col, target_vals):
    return ['{}_coverted_exp_for_{}={}'.format(col, target_col, v) for v in target_vals]


def process_with_lambda(train_df, test_df, col, target_col, target_vals, lambda_f):
    temp_target = '{}_'.format(target_col)
    train_df[temp_target]= train_df[target_col]
    train_df= pd.get_dummies(train_df, columns=[target_col])
    dummies_cols = [dummy_col(target_col, v) for v in target_vals]
    priors = train_df[dummies_cols].mean()
    priors_arr = [priors[dummy_col(target_col, v)] for v in target_vals]
    agg = OrderedDict(
        [(dummy_col(target_col, v), OrderedDict([('{}_mean'.format(v),'mean'),('{}_count'.format(v),'count')])) for v in target_vals]
    )
    df = train_df[[col]+dummies_cols].groupby(col).agg(agg)
    cols_blja = []
    for v in target_vals:
        cols_blja.append('posterior_{}'.format(v))
        cols_blja.append('count_{}'.format(v))
    df.columns = cols_blja
    new_cols=[]
    for v in target_vals:
        def norm_posterior(x):
            cnt = x['count_{}'.format(v)]
            cnt= float(cnt)
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


def processM(train_df, test_df, col, target_col, target_vals, m):
    m = float(m)
    temp_target = '{}_'.format(target_col)
    train_df[temp_target]= train_df[target_col]
    train_df= pd.get_dummies(train_df, columns=[target_col])
    dummies_cols = [dummy_col(target_col, v) for v in target_vals]
    priors = train_df[dummies_cols].mean()
    priors_arr = [priors[dummy_col(target_col, v)] for v in target_vals]
    agg = OrderedDict(
        [(dummy_col(target_col, v), OrderedDict([('{}_mean'.format(v),'mean'),('{}_count'.format(v),'count')])) for v in target_vals]
    )
    df = train_df[[col]+dummies_cols].groupby(col).agg(agg)
    cols_blja = []
    for v in target_vals:
        cols_blja.append('posterior_{}'.format(v))
        cols_blja.append('count_{}'.format(v))
    df.columns = cols_blja
    new_cols=[]
    for v in target_vals:
        def norm_posterior(x):
            cnt = x['count_{}'.format(v)]
            cnt= float(cnt)
            posterior = x['posterior_{}'.format(v)]
            prior = priors[dummy_col(target_col, v)]
            l = cnt/(cnt+m)
            return (l * posterior) + ((1 - l) * prior)

        new_col = '{}_catM={}_for_{}={}'.format(col, m, target_col, v)
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

def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)


def get_exp_lambda(k,f):
    def res(n):
        return 1/(1+math.exp(float(k-n)/f))
    return res

def visualize_exp_lambda():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sz=300
    ixes = range(sz)
    l0=get_exp_lambda(50, 1)
    l1=get_exp_lambda(50, 5)
    l2=get_exp_lambda(50, 0.2)
    to_plot0 = [l0(x) for x in ixes]
    to_plot1 = [l1(x) for x in ixes]
    to_plot2 = [l2(x) for x in ixes]
    ax.plot(ixes, to_plot0, label='f=1')
    ax.plot(ixes, to_plot1, label='f=5')
    ax.plot(ixes, to_plot2, label='f=0.2')
    ax.legend()