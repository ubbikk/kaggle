from fuzzywuzzy.fuzz import QRatio, WRatio, \
    partial_ratio, partial_token_set_ratio, partial_token_sort_ratio, \
    token_set_ratio, token_sort_ratio

# same length
from scipy.spatial.distance import cosine, cityblock, canberra, euclidean, minkowski, seuclidean, \
    braycurtis, chebyshev, correlation, mahalanobis

from scipy.stats import skew, kurtosis

# boolean

from scipy.spatial.distance import dice, kulsinski, jaccard, \
    rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# sets
import distance
# from distance import levenshtein, sorensen, jaccard, nlevenshtein

BOOL_METRICS = {x.__name__: x for x in
                [dice, kulsinski, jaccard]}
FUZZY_METRICS = {x.__name__: x for x in
                 [QRatio, WRatio, partial_ratio, partial_token_set_ratio,
                  partial_token_sort_ratio, token_set_ratio, token_sort_ratio]}

REAL_METRICS = {x.__name__: x for x in
                [cosine, cityblock, canberra, euclidean, seuclidean,
                 braycurtis, chebyshev, correlation, mahalanobis]}
REAL_METRICS['minkowski_3']=lambda x,y: minkowski(x,y, 3)
REAL_STATISTICS={x.__name__: x for x in [skew, kurtosis]}

SEQUENCES_METRICS = {x.__name__: x for x in
                [distance.levenshtein, distance.sorensen, distance.nlevenshtein]}
SEQUENCES_METRICS['distance.jaccard']=distance.jaccard


def generate_bool_vectors(a, b):
    a=set(a)
    b=set(b)
    join = set(a)
    join.update(b)
    join=list(join)
    join.sort()
    return [x in a for x in join], [x in b for x in join]




def process_sets_metrics(df, col1, col2, prefix, fp):
    df['bool_vectors'] = df[[col1, col2]].apply(lambda s: generate_bool_vectors(s[col1].split(), s[col2].split()), axis=1)
    new_cols = []
    for name, func in BOOL_METRICS.iteritems():
        print name
        new_col = '{}_{}'.format(prefix, name)
        new_cols.append(new_col)
        def wrap_func(s):
            try:
                return func(s[0], s[1])
            except:
                print '{}=null'.format(name)
                return -1

        df[new_col] = df['bool_vectors'].apply(wrap_func)

    df[new_cols].to_csv(fp, index_label='id')


def process_fuzzy_metrics(df, col1, col2, prefix, fp):
    new_cols = []
    for name, func in FUZZY_METRICS.iteritems():
        print name
        new_col = '{}_{}'.format(prefix, name)
        new_cols.append(new_col)
        df[new_col] = df[[col1, col2]].apply(lambda s: func(s[col1],s[col2]), axis=1)

    df[new_cols].to_csv(fp, index_label='id')

def process_sequence_metrics(df, col1, col2, prefix, fp):
    new_cols = []
    for name, func in SEQUENCES_METRICS.iteritems():
        print name
        new_col = '{}_{}'.format(prefix, name)
        new_cols.append(new_col)
        df[new_col] = df[[col1, col2]].apply(lambda s: func(s[col1],s[col2]), axis=1)

    df[new_cols].to_csv(fp, index_label='id')



    # from distance import hamming # the same length
    # from distance import lcsubstrings #Find the longest common substring(s)


    # - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
    #                       'manhattan']. These metrics support sparse matrix inputs.
    #
    # - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
    #                                 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
    #                                 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    #                                 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    # See the documentation for scipy.spatial.distance for details on these
    # metrics. These metrics do not support sparse matrix inputs.
