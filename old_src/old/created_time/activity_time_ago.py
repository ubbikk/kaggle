import pandas as pd

MANAGER_ID = 'manager_id'
CREATED = 'created'

def process_recent_activity(train_df, test_df):
    train_test_df = pd.concat([train_df, test_df])
    x, y = get_activity_in_sec_cols(train_test_df)

    past_col = 'past_manager_activity'
    future_col = 'future_manager_activity'
    test_df[past_col] = x[test_df.index]
    test_df[future_col] = y[test_df.index]

    x, y = get_activity_in_sec_cols(train_df)
    train_df[past_col] = x[train_df.index]
    train_df[future_col] = y[train_df.index]

    return train_df, test_df, [past_col, future_col]

def get_activity_in_sec_cols(df):
    groups= df.groupby(MANAGER_ID)[CREATED].groups
    for k,v in groups.iteritems():
        v = df.loc[v, CREATED].tolist()
        v.sort()
        groups[k] = v

    def find_diff_past(s):
        g = groups[s[MANAGER_ID]]
        sz=len(g)
        if sz==1:
            return -1

        created = s[CREATED]
        ind = g.index(created)

        if ind==0:#first
            if g[1]==created:
                return 0
            return -1

        if ind==sz-1:#last
            return (g[ind]-g[ind-1]).total_seconds()

        return (g[ind]-g[ind-1]).total_seconds()

    def find_diff_future(s):
        g = groups[s[MANAGER_ID]]
        sz=len(g)
        if sz==1:
            return -1

        created = s[CREATED]
        ind = g.index(created)

        if ind==0:#first
            return (g[ind+1]-g[ind]).total_seconds()

        if ind==sz-1:#last
            if g[ind-1]==created:
                return 0
            return -1

        return (g[ind+1]-g[ind]).total_seconds()

    return df.apply(find_diff_past, axis=1), df.apply(find_diff_future, axis=1)