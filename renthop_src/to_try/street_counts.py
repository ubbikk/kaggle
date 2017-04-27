import pandas as pd

DISPLAY_ADDRESS = 'display_address'
NORMALIZED_DISPLAY_ADDRESS = 'normalized_display_address'
MANAGER_ID = 'manager_id'

def reverse_norm_map(m):
    res = {}
    for k, v in m.iteritems():
        for s in v:
            res[s.lower()] = k.lower()

    return res


NORMALIZATION_MAP = {
    'street': ['St', 'St.', 'Street', 'St,', 'st..', 'street.'],
    'avenue': ['Avenue', 'Ave', 'Ave.'],
    'square': ['Square'],
    'east': ['e', 'east', 'e.'],
    'west': ['w', 'west', 'w.'],
    'road':['road', 'rd', 'rd.']
}

REVERSE_NORM_MAP = reverse_norm_map(NORMALIZATION_MAP)


# Fifth, Third

def normalize_tokens(s):
    tokens = s.split()
    for i in range(len(tokens)):
        tokens[i] = if_starts_with_digit_return_digit_prefix(tokens[i])
        t = tokens[i]
        if t.lower() in REVERSE_NORM_MAP:
            tokens[i] = REVERSE_NORM_MAP[t.lower()]
    return ' '.join(tokens)

def if_starts_with_digit_return_digit_prefix(s):
    if not s[0].isdigit():
        return s
    last=0
    for i in range(len(s)):
        if s[i].isdigit():
            last=i
        else:
            break

    return s[0:last+1]


def normalize_string(s):
    s = normalize_tokens(s)
    if s == '':
        return s

    s=s.lower()

    tokens = s.split()
    if len(tokens) == 2:
        return ' '.join(tokens)
    if tokens[0].replace('.', '').replace('-', '').isdigit():
        return ' '.join(tokens[1:])
    else:
        return ' '.join(tokens)

def normalize_display_address_df(df):
    df[NORMALIZED_DISPLAY_ADDRESS] = df[DISPLAY_ADDRESS].apply(normalize_string)

def process_street_counts(train_df, test_df):
    df = pd.concat([train_df, test_df])
    normalize_display_address_df(df)
    col = 'street_popularity'
    df[col] = df.groupby(NORMALIZED_DISPLAY_ADDRESS)[MANAGER_ID].transform('count')

    train_df[col]=df.loc[train_df.index, col]
    test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, [col]

