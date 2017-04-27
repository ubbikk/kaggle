MONTH_FREE = 'month_free'
DESCRIPTION = 'description'

def is_month_free(s):
    if s is None:
        return False
    return ('months free' in s.lower()) or ('month free' in s.lower())


def process_month_free(train_df, test_df):
    for df in[train_df, test_df]:
         df[MONTH_FREE] = df[DESCRIPTION].apply(is_month_free)

    return train_df, test_df, [MONTH_FREE]