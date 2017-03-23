import re

import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

F_COL=u'features'
DESCRIPTION= 'description'

BAD_SYMBOLS_REGEX= re.compile('[\.*-+&$#,\()\{\}\'~!?:;"\<\>]')
HTML_TOKENS_PATTERN = re.compile('<br/>|<br>|<div>|<p>|</p>')
BAD_TOKEN_PATTERN = re.compile('.*\d.*')
BAD_TOKENS= {'\\', '-', '/'}

TOKENIZER = TweetTokenizer()


def lower_df(df):
    df[DESCRIPTION]=df[DESCRIPTION].apply(lambda l: l.lower())

def normalize_df(df):
    # lower_df(df)
    df[DESCRIPTION]=df[DESCRIPTION].apply(lambda s: HTML_TOKENS_PATTERN.sub(' ', s))
    df[DESCRIPTION]=df[DESCRIPTION].apply(lambda s: BAD_SYMBOLS_REGEX.sub(' ', s))

def get_c_map_tokens(df, N=None):
    normalize_df(df)
    c_map={}
    for l in df[DESCRIPTION]:
        for t in TOKENIZER.tokenize(l):
            if t in c_map:
                c_map[t]+=1
            else:
                c_map[t]=1


    c_map=[(k,v) for k,v in c_map.iteritems()]
    c_map.sort(key=lambda s:s[1], reverse=True)

    if N is None:
        N=len(c_map)

    return c_map[:N]

def get_c_map_tokens_no_stop_words(df, N=None):
    c_map = get_c_map_tokens(df)
    c_map = [x for x in c_map if x[0].lower() not in STOP_WORDS]
    if N is None:
        N=len(c_map)

    return c_map[:N]

def get_c_map_tokens_upper_words(df, N=None):
    c_map = get_c_map_tokens_no_stop_words(df)
    def is_upper(s):
        if len(s)<2:
            return False
        return s[0].isupper() and (not s[1].isupper())
    return [x for x in c_map if is_upper(x[0])]