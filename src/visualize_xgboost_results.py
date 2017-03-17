import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns

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

def get_res_at_N(res, N):
    return [x[N] for x in res]


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