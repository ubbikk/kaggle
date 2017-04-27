import re

import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

F_COL=u'features'

BAD_SYMBOLS_REGEX= re.compile('[\.*-+&$#,\()\{\}\'~!?:;"\<\>]')
BAD_TOKEN_PATTERN = re.compile('.*\d.*')
BAD_TOKENS= {'\\', '-', '/'}

TOKENIZER = TweetTokenizer()

#TODO
#u'central a/c'


def lower_df(df):
    df[F_COL]=df[F_COL].apply(lambda l: [x.lower() for x in l])

def normalize_feature(s):
    s=s.lower()
    return BAD_SYMBOLS_REGEX.sub(' ', s)

def should_skip_token(s):
    if s in STOP_WORDS:
        return True

    if s in BAD_TOKENS:
        return True

    return re.match(BAD_TOKEN_PATTERN, s) is not None

def normalize_token_df(df):
    def l_to_tokens_list(l):
        s = ' '.join(l)
        s= normalize_feature(s)
        return TOKENIZER.tokenize(s)

    df[F_COL] = df[F_COL].apply(l_to_tokens_list)

    return df


def get_c_map_tokens(df, N=None):
    lower_df(df)
    tokenizer = TweetTokenizer()
    c_map={}
    for l in df[F_COL]:
        for f in l:
            f=normalize_feature(f)
            for t in tokenizer.tokenize(f):
                if should_skip_token(t):
                    continue
                if t in c_map:
                    c_map[t]+=1
                else:
                    c_map[t]=1


    c_map=[(k,v) for k,v in c_map.iteritems()]
    c_map.sort(key=lambda s:s[1], reverse=True)

    if N is None:
        N=len(c_map)

    return c_map[:N]

def get_top_N_counts_tokens(df, N=None):
    c_map_ordered = get_c_map_tokens(df)
    if N is None:
        N = len(c_map_ordered)
    return c_map_ordered[:N]

def get_top_N_tokens(df,N):
    return [x[0] for x in get_top_N_counts_tokens(df, N)]

def add_top_N_tokens_df(df, N):
    top_N = get_top_N_tokens(df, N)
    normalize_token_df(df)
    col_to_series={}
    new_cols=[]
    for f in top_N:
        s = df[F_COL].apply(lambda l: 1 if f in l else 0)
        new_col = val_to_col(f)
        col_to_series[new_col] = s
        new_cols.append(new_col)

    for col in df.columns.values:
        col_to_series[col] = df[col]

    return pd.DataFrame(col_to_series), new_cols

def add_tokens_list(df, l):
    normalize_token_df(df)
    for col in l:
        f = col_to_val(col)
        df[col] = df[F_COL].apply(lambda l: 1 if f in l else 0)

    return df

def val_to_col(s):
    return s.replace(' ', '_') + '_'

def col_to_val(col):
    return col.replace('_', ' ').strip()


def write_tokens_counts(df, fp):
    lower_df(df)
    tokenizer = TweetTokenizer()
    c_map={}
    for l in df[F_COL]:
        for f in l:
            for t in tokenizer.tokenize(f):
                if t in c_map:
                    c_map[t]+=1
                else:
                    c_map[t]=1


    c_map=[(k,v) for k,v in c_map.iteritems()]
    c_map.sort(key=lambda s:s[1], reverse=True)
    with open(fp, 'w+') as f:
        f.write('\n'.join([str(x) for x in c_map]))


def write_tokens_counts_normalized1(df, fp):
    c_map = get_c_map_tokens(df)
    with open(fp, 'w+') as f:
        f.write('\n'.join([str(x) for x in c_map]))
