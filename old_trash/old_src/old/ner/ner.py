import re

from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import TweetTokenizer

stanford_classifier = '/home/dpetrovskyi/stanford/stanford_nlp_libs/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz'
stanford_ner_path = '/home/dpetrovskyi/stanford/stanford_nlp_libs/stanford-ner-2015-12-09/stanford-ner.jar'
TAGGER = StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')
TOKENIZER=TweetTokenizer()
DESCRIPTION= 'description'
class Counter:
    def __init__(self):
        self.c=0

    def increment(self):
        self.c+=1
        print self.c

HTML_TOKENS_PATTERN = re.compile('<br/>|<br>|<div>|<p>|</p>')



def perform_ner(s, counter):
    counter.increment()
    tokens= TOKENIZER.tokenize(s)
    tags = TAGGER.tag(tokens)
    print tags
    print

    start = 0
    res=[]
    while start<len(tags):
        t = tags[start]
        val = t[1]
        if val=='O':
            start+=1
            continue

        end = start+1
        while end <len(tags):
            if tags[end][1] != val:
                break
            end+=1

        res.append(tags[start:end])
        start=end

    res_new=[]
    for x in res:
        tag = x[0][1]
        val = ' '.join([y[0] for y in x])
        res_new.append((val, tag))

    return res_new



def process_df(df):
    counter = Counter()
    df['descr_normalized']=df[DESCRIPTION].apply(lambda s: HTML_TOKENS_PATTERN.sub(' ', s))
    df['ner'] = df['descr_normalized'].apply(lambda s: perform_ner(s, counter))