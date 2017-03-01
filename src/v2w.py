from __future__ import print_function
import gensim
from gensim.models import word2vec
import os.path
import csv
from storm.drpc import DRPCClient
import json
import numpy as np
import pandas as pd
import random as rnd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.linear_model import LinearRegression
from scipy.sparse import vstack

__tokenizer__ = TweetTokenizer()
__sentence_splitter__ = PunktSentenceTokenizer()

__bad_token__ = re.compile('[.;,-?()\'"]')


def weighted_avg_vector_for_text(text, model, def_len=300, delta=0.01):
    vectors =[]
    for t in tokenize_v2w(text):
        try:
            v = model[t]
            vectors.append(v)
        except KeyError:
            continue

    if len(vectors)==0:
        vectors.append(np.zeros(def_len,))
        vectors.append(np.zeros(def_len,))

    vectors = [vectors[j]*(1+(delta*j)) for j in range(len(vectors))]

    vstack = np.vstack(vectors)
    return np.mean(vstack, axis=0)


def avg_vector_df(df, model, col_name):
    vector_len = 300
    m = list_of_text_to_features_matrix(df[col_name], model)
    cols=['word_2_vec_{}'.format(i) for i in range(vector_len)]
    new_df = pd.DataFrame(m, index=df.index, columns=cols)

    return pd.merge(df, new_df, left_index=True, right_index=True), cols

def avg_vector_df_and_pca(train_df, test_df,  model, col_name, pca_dim=10):
    vector_len = 300
    pca = PCA(n_components=pca_dim)

    matrix_train = list_of_text_to_features_matrix(train_df[col_name], model)
    matrix_test = list_of_text_to_features_matrix(test_df[col_name], model)

    matrix_train = pca.fit_transform(matrix_train)
    matrix_test = pca.transform(matrix_test)

    cols=['word_2_vec_{}'.format(i) for i in range(pca_dim)]

    new_train_df = pd.DataFrame(matrix_train, index=train_df.index, columns=cols)
    new_test_df = pd.DataFrame(matrix_test, index=test_df.index, columns=cols)

    return pd.merge(train_df, new_train_df, left_index=True, right_index=True), pd.merge(test_df, new_test_df, left_index=True, right_index=True), cols



def avg_vector_for_text(text, model, def_len=300):
    # print(text)a
    vectors =[]
    for t in tokenize_v2w(text):
        try:
            v = model[t]
            vectors.append(v)
        except KeyError:
            continue

    if len(vectors)==0:
        vectors.append(np.zeros(def_len,))
        vectors.append(np.zeros(def_len,))

    vstack = np.vstack(vectors)
    return np.mean(vstack, axis=0)


def tokenize_v2w(text):
    res = __tokenizer__.tokenize(text)
    res = filter(lambda s: not __bad_token__.match(s), res)
    return res



def list_of_text_to_features_matrix(list_of_text, model):
    features =[]
    for t in list_of_text:
        v = avg_vector_for_text(t, model, def_len=300)
        # print(v.shape)
        features.append(v)
    if len(list_of_text)==0:
        features.append(np.zeros(300,))

    return np.vstack(features)

def list_of_text_to_features_matrix_weighted(list_of_text, model, delta=0.01):
    features =[]
    for t in list_of_text:
        v = weighted_avg_vector_for_text(t, model, def_len=300, delta=delta)
        # print(v.shape)
        features.append(v)

    return np.vstack(features)

def split_into_sentences_and_convert_to_features_matrix(text, model):
    try:
        sentences = __sentence_splitter__.tokenize(text)
    except:
        print(text)
        sentences=[]
    return list_of_text_to_features_matrix(sentences, model)


def predict_sentence_wise(texts, classifier, model):
    r = []
    for t in texts:
        res = classifier.predict(split_into_sentences_and_convert_to_features_matrix(t, model))
        res = np.sum(res)
        res = 1 if res>=0 else -1
        r.append(res)

    return np.array(r)


def load_model():
    return gensim.models.Word2Vec.load_word2vec_format('/home/dpetrovskyi/GoogleNews-vectors-negative300.bin', binary=True)
