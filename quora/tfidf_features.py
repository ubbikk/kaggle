import pandas as pd
import numpy as np
import seaborn as sns
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words_with_PRON = set(stop_words)
stop_words_with_PRON.add('PRON')

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

fp_train = '../../data/train.csv'
fp_test = '../../data/test.csv'

lemmas_train_fp = '../../data/train_lemmas.csv'
tokens_train_fp = '../../data/train_tokens.csv'
nlp_train_fp = '../../data/postag_ner_train.json'

stems_train_fp = '../../data/train_porter.csv'
stems_test_fp = '../../data/test_porter.csv'

normalized_train_fp = '../../data/train_normalized.csv'
common_words_train_fp = '../../data/train_common_words.csv'
length_train_fp = '../../data/train_length.csv'

METRICS_FP = [
    '../../data/train_metrics_bool_lemmas.csv',
    '../../data/train_metrics_bool_stems.csv',
    '../../data/train_metrics_bool_tokens.csv',
    '../../data/train_metrics_fuzzy_lemmas.csv',
    '../../data/train_metrics_fuzzy_stems.csv',
    '../../data/train_metrics_fuzzy_tokens.csv',
    '../../data/train_metrics_sequence_lemmas.csv',
    '../../data/train_metrics_sequence_stems.csv',
    '../../data/train_metrics_sequence_tokens.csv'
]

TARGET = 'is_duplicate'
qid1, qid2 = 'qid1', 'qid2'

question1, question2 = 'question1', 'question2'
lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'
stems_q1, stems_q2 = 'stems_q1', 'stems_q2'
tokens_q1, tokens_q2 = 'tokens_q1', 'tokens_q2'

def load_train():
    return pd.read_csv(fp_train, index_col='id')


def load__train_metrics():
    dfs = [pd.read_csv(fp, index_col='id') for fp in METRICS_FP]
    return pd.concat(dfs, axis=1)


def load_train_all():
    return pd.concat([
        load_train(),
        load_train_lemmas(),
        load_train_stems(),
        load_train_tokens(),
        load_train_lengths(),
        load_train_common_words(),
        load__train_metrics()
    ], axis=1)


def load_train_test():
    return pd.read_csv(fp_train, index_col='id'), pd.read_csv(fp_test, index_col='test_id')


def load_train_lemmas():
    df = pd.read_csv(lemmas_train_fp, index_col='id')
    df = df.fillna('')
    return df


def load_train_tokens():
    df = pd.read_csv(tokens_train_fp, index_col='id')
    df = df.fillna('')
    return df


def load_train_stems():
    df = pd.read_csv(stems_train_fp, index_col='id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter': 'stems_q1', 'question2_porter': 'stems_q2'})
    df = df.fillna('')
    return df


def load_train_common_words():
    df = pd.read_csv(common_words_train_fp, index_col='id')
    return df


def load_train_lengths():
    df = pd.read_csv(length_train_fp, index_col='id')
    return df


def load_train_normalized_train():
    return pd.read_csv(normalized_train_fp, index_col='id')



def write_tfidf_features(train_df, test_df, col1, col2, prefix, stopwords, fp):
    tfidf = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 1))
    for col in [col1, col2]:
        for df in [train_df, test_df]:
            df[col].fillna('', inplace=True)
    #sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
    def convert_series(s):
        return s[~s.isnull()].apply(lambda s: s.lower()).tolist()

    bl = [train_df[col1], train_df[col2], test_df[col1], test_df[col2]]
    corpus=[]
    for x in bl:
        corpus+=convert_series(x)
    print len(corpus)
    tfidf.fit_transform(corpus)

    for i, df in enumerate([train_df, test_df]):
        label = 'train' if i==0 else 'test'
        index_label = 'id' if i==0 else 'test_id'
        new_cols=[]

        new_col = '{}_tfidf_mean_q1'.format(prefix)
        new_cols.append(new_col)
        df[new_col] = df[col1].apply(lambda s: np.mean(tfidf.transform([s])).data)

        new_col = '{}_tfidf_mean_q2'.format(prefix)
        new_cols.append(new_col)
        df[new_col] = df[col2].apply(lambda s: np.mean(tfidf.transform([s])).data)

        new_col = '{}_tfidf_sum_q1'.format(prefix)
        new_cols.append(new_col)
        df[new_col] = df[col1].apply(lambda s: np.sum(tfidf.transform([s])).data)

        new_col = '{}_tfidf_sum_q2'.format(prefix)
        new_cols.append(new_col)
        df[new_col] = df[col2].apply(lambda s: np.sum(tfidf.transform([s])).data)



        df = df[new_cols]
        df.to_csv(os.path.join(fp, '{}_tfidf_{}'.format(prefix, label)), index_label=index_label)



    # idf = tfidf.idf_
    # stats= dict(zip(tfidf.get_feature_names(), idf))



