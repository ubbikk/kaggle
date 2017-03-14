import pandas as pd
from collections import OrderedDict
import math
import seaborn as sns
sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)


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


def processM(train_df, test_df, col, target_col, target_vals, m):
    lambda_f = lambda cnt: cnt/(cnt+m)
    return process_with_lambda(train_df, test_df, col, target_col, target_vals, lambda_f)

def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)


def get_exp_lambda(k,f):
    def res(n):
        return 1/(1+math.exp(float(k-n)/f))
    return res


def visualize_exp_lambda(k,f):
    import matplotlib.pyplot as plt
    plt.interactive(False)

    fig, ax = plt.subplots()
    sz=300
    ixes = range(sz)
    l0=get_exp_lambda(k, f)
    to_plot0 = [l0(x) for x in ixes]
    ax.plot(ixes, to_plot0, label='')
    ax.legend()
    plt.show()


def create_count_label_column(df, count_col, new_col, cutoffs):
    def count_lbl(x):
        if x<=cutoffs[0]:
            return '(0, {}]'.format(cutoffs[0])
        sz = len(cutoffs)
        for i in range(sz -1):
            start = cutoffs[i]
            end = cutoffs[i+1]
            if x>start and x<=end:
                if end-start>1:
                    return '({}, {}]'.format(start, end)
                else:
                    return str(end)

        last = cutoffs[sz - 1]
        if x> last:
            return '({}, inf)'.format(last)

        raise Exception()

    df[new_col]=df[count_col].apply(count_lbl)
    return df

def cutoffs_labels(cutoffs):
    first = cutoffs[0]
    sz=len(cutoffs)
    last=cutoffs[sz-1]
    res = []
    res.append('(0, {}]'.format(first))
    for i in range(sz-1):
        start = cutoffs[i]
        end = cutoffs[i+1]
        if end-start>1:
            res.append('({}, {}]'.format(start, end))
        else:
            res.append(str(end))

    res.append('({}, inf)'.format(last))

    return res


def explore_value_counts(df, col_name, cutoffs):
    tmp=df[col_name].to_frame(col_name)
    tmp['count_col'] = df.groupby(col_name)[col_name].transform('count')
    tmp = create_count_label_column(tmp, 'count_col', 'count_label', cutoffs)
    sns.factorplot(x='count_label', data=tmp, kind='count', order=cutoffs_labels(cutoffs))



# def visualize_exp_lambda():
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots()
#     sz=300
#     ixes = range(sz)
#     l0=get_exp_lambda(50, 1)
#     l1=get_exp_lambda(50, 5)
#     l2=get_exp_lambda(50, 0.2)
#     to_plot0 = [l0(x) for x in ixes]
#     to_plot1 = [l1(x) for x in ixes]
#     to_plot2 = [l2(x) for x in ixes]
#     ax.plot(ixes, to_plot0, label='f=1')
#     ax.plot(ixes, to_plot1, label='f=5')
#     ax.plot(ixes, to_plot2, label='f=0.2')
#     ax.legend()