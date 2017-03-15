import json
import numpy as np

def do_work():
    res = json.load(open('/home/dpetrovskyi/PycharmProjects/kaggle/src/bid_mngr_and_earlystopping/statistics_on_n_estimators'))
    res = [x['test'] for x in res]

    l = [np.mean([x[j] for x in res]) for j in range(1000)]
