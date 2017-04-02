import pandas as pd

MANAGER_ID = 'manager_id'
LABEL='lbl'

def process_topN_managers(train_df, test_df):
    N=10
    new_cols=[]
    df = pd.concat([train_df, test_df])
    top_mngrs=df.groupby(MANAGER_ID)[MANAGER_ID].count().sort_values(ascending=False).index.values[:N]
    for m in top_mngrs:
        new_col = 'mngr_{}'.format(m)
        new_cols.append(new_col)
        for df in [train_df, test_df]:
            df[new_col] = df[MANAGER_ID].apply(lambda s: 1 if s==m else 0)

    return train_df, test_df, new_cols
