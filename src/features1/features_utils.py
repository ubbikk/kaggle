import pandas as pd

F_COL=u'features'

def lower_df(df):
    df[F_COL]=df[F_COL].apply(lambda l: [x.lower() for x in l])

def get_c_map(s):
    s=s.apply(lambda l: [x.lower() for x in l])
    c_map = {}
    for l in s:
        for x in l:
            if x in c_map:
                c_map[x]+=1
            else:
                c_map[x]=1

    return c_map

def get_c_map_ordered(s):
    c_map = get_c_map(s)
    c_map=[(k,v) for k,v in c_map.iteritems()]
    c_map.sort(key=lambda s:s[1], reverse=True)

    return c_map

def get_top_N_counts(s,N):
    return get_c_map_ordered(s)[:N]

def get_top_N_features(s,N):
    return [x[0] for x in get_top_N_counts(s, N)]


def add_top_N_features_df(df, N):
    df[F_COL]= df[F_COL].apply(lambda l: [x.lower() for x in l])
    top_N = get_top_N_features(df[F_COL], N)
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

def val_to_col(s):
    return s.replace(' ', '_') + '_'

def col_to_val(col):
    return col.replace('_', ' ').strip()


def add_features_list(df, l):
    lower_df(df)
    for col in l:
        f = col_to_val(col)
        df[col] = df[F_COL].apply(lambda l: 1 if f in l else 0)

    return df

