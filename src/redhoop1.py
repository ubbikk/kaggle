import json
import os
import seaborn as sns
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np

sns.set(color_codes=True)
os.chdir('/home/dpetrovskyi/PycharmProjects/kaggle/src')

train_file = '../data/redhoop/train.json'
test_file = '../data/redhoop/test.json'


def load_train():
    return pd.read_json(train_file)

def load_test():
    return pd.read_json(test_file)


