import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import math

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

def plot_errors_one_run(estimator):
    results = estimator.evals_result()
    fig, ax = plt.subplots()
    to_plot0 = results['validation_0']['mlogloss']
    to_plot1 = results['validation_1']['mlogloss']
    x = range(len(to_plot0))
    ax.plot(x, to_plot0, label='train')
    ax.plot(x, to_plot1, label='test')
    ax.legend()

def load_fp(fp):
    with open(fp)as f:
        return json.load(f)

def get_res_at_N_arr(res, N):
    return [x[N] for x in res]

def get_test_res(res):
    return [x['test'] for x in res]

def get_res_at_N_fp(fp, N):
    return get_res_at_N_arr(get_test_res(load_fp(fp)), N)


def explore_res_fp_xg_Ns(fp, Ns=(1000,)):
    res_test=[x['test'] for x in load_fp(fp)]
    sz = len(res_test)
    print 'Num {}'.format(sz)
    res1K = get_res_at_N_arr(res_test, 1000)
    std = np.std(res1K)
    std = std/math.sqrt(sz)
    print 'std of avg_los {}'.format(std)
    print
    for N in Ns:
        res_N = get_res_at_N_arr(res_test, N)
        print 'N {}'.format(N)
        print 'avg_loss {}'.format(np.mean(res_N))
        print 'std {}'.format(np.std(res_N))
        print



def plot_errors_fp(fp):
    with open(fp)as f:
        results = json.load(f)
        train_runs= [x['train'] for x in results]
        test_runs= [x['test'] for x in results]

        sz=len(train_runs[0])
        x_axis=range(sz)
        y_train = [np.mean([x[j] for x in train_runs]) for j in x_axis]
        y_test = [np.mean([x[j] for x in test_runs]) for j in x_axis]

        fig, ax = plt.subplots()
        ax.plot(x_axis, y_train, label='train')
        ax.plot(x_axis, y_test, label='test')
        ax.legend()


# fp='/home/dpetrovskyi/PycharmProjects/kaggle/src/bath_bedrooms_features/results_1500.json.json'
# plot_errors_fp(fp)