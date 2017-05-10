import re
import pandas as pd

pattern = re.compile('.+\d{3}-\d{3}-\d{4}')
DESCRIPTION = 'description'


def process_phone_number(train_df, test_df):
    df = pd.concat([train_df, test_df])
    new_cols = []

    def has_phone(s):
        if re.match(pattern, s) is not None:
            return 1
        return 0

    new_col = 'has_phone_number'
    df[new_col] = df[DESCRIPTION].apply(has_phone)
    print 'has phone {}'.format(len(df[df[new_col]==1]))
    new_cols.append(new_col)

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
    test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols