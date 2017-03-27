from pymongo import MongoClient
import numpy as np
from numpy import mean, std
from scipy.stats import ttest_ind
client = MongoClient('10.20.0.144', 27017)
db = client.renthop_results

def load_results(name):
    collection = db[name]
    return [x['results'] for x in collection.find()]

def load_results_1K(name):
    collection = db[name]
    return [x['results']['test'][1000] for x in collection.find()]