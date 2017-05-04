import spacy

nlp = spacy.load('en', parser=False)

txt = u'How did Darth Vader fought Darth Maul in Star Wars Legends?'
doc = nlp(txt)

for word in doc:
    print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)

for e in doc.ents:
    print e.start, e.end, e.label_, str(e)


counter=0

def process_df(df, cols):
    def postag_and_ner(s):
        global counter
        counter+=1
        if counter%1000 == 0:
            print counter
        doc=nlp(str(s).decode("utf-8"))
        pos = [[word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_] for word in doc]
        ner = [[e.start, e.end, e.label_, str(e)] for e in doc.ents]
        return [pos, ner]
    for col in cols:
        df['nlp_{}'.format(col)]=df[col].apply(postag_and_ner)
    df.to_json('processed.json')

