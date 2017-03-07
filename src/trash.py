def most_frequent(l):
    m = {}
    for x in l:
        if x in m:
            m[x]+=1
        else:
            m[x]=1

    m = [(k,v) for k,v in m.iteritems()]
    m.sort(key=lambda s: s[1], reverse=True)
    return m[0][0]

