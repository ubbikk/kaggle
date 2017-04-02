import pandas as pd

NEI = 'neighbourhood'
BORO = 'boro'
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'
BATHROOMS = 'bathrooms'
BEDROOMS = 'bedrooms'
BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'
PRICE = 'price'

import json
import pandas as pd

rent_file = '/home/dpetrovskyi/PycharmProjects/kaggle/src/with_geo/data/neis_from_renthop_lower.json'
TARGET = 'interest_level'

EXACT_MAP = {
    'gramercy': 'gramercy park',
    'clinton': "hell's kitchen",
    'turtle bay': 'midtown east',
    'tudor city': 'midtown east',
    'sutton place': 'midtown east',
    'hamilton heights': 'west harlem',
    'bedford stuyvesant': 'bedford-stuyvesant',
    'hunters point': 'long island city',
    'battery park': 'battery park city',
    'manhattanville': 'west harlem',
    'carnegie hill': 'upper east side',
    'stuyvesant town': 'stuyvesant town - peter cooper village',
    'downtown': 'downtown brooklyn',
    'morningside heights': 'west harlem',
    'spuyten duyvil': 'riverdale',
    'prospect lefferts gardens': 'flatbush',
    'greenwood': 'greenwood heights',
    'fort hamilton': 'bay ridge',
    'high bridge': 'highbridge',
    'columbia street waterfront district': 'carroll gardens',
    'ocean parkway': 'midwood',
    'north riverdale': 'riverdale',
    'astoria heights': 'astoria',
    'tremont': 'mount hope',
    'homecrest': 'sheepshead bay',
    'new utrecht': 'borough park',
    'fieldston': 'riverdale',
    'georgetown': 'upper east side',
    'tottenville': 'washington heights',
    'hillcrest': 'kew gardens hills',
    'oakland gardens': 'forest hills',
    'pomonok': 'washington heights',
    'wingate': 'east flatbush',
    'fordham': 'fordham manor',
    'forest hills gardens': 'forest hills',
    'columbus circle': "hell's kitchen"
}

SPECIAL = {
    'midtown': ('midtown', 'midtown manhattan', 'manhattan'),
    'harlem': ('harlem', 'upper manhattan', 'manhattan')
}

ONLY_SECOND = {
    'castle hill': ('2nd', 'east bronx', 'bronx'),
    'throggs neck': ('2nd', 'east bronx', 'bronx'),
    'soundview': ('2nd', 'east bronx', 'bronx'),
    'port morris': ('2nd', 'east bronx', 'bronx'),
}

ONLY_THIRD = {
    'queens village': ('3rd', '3rd', 'queens'),
    'laurelton': ('3rd', '3rd', 'queens')
}


def load_rent():
    m = json.load(open(rent_file))
    res = {}
    for boro, boro_m in m.iteritems():
        for sub_boro, neis in boro_m.iteritems():
            for n in neis:
                res[n] = [n, sub_boro, boro]

    return res

def explore_target_on_val(df, col, val):
    return pd.get_dummies(df[[TARGET, col]][df[col] == val], columns=[TARGET]).mean()


def transform_geo_to_rent(s):
    if s is None:
        return s
    s=s.lower()
    rent = load_rent()
    if s in rent:
        return rent[s]

    if s in EXACT_MAP:
        return rent[EXACT_MAP[s]]

    if s in SPECIAL:
        return SPECIAL[s]

    return ('not_mapped_yet', 'not_mapped_yet', 'not_mapped_yet')


def dummy_col(col_name, val):
    return '{}_{}'.format(col_name, val)

def get_dummy_cols(col_name, col_values):
    return ['{}_{}'.format(col_name, val) for val in col_values]


def normalize_bed_bath(df):
    df[BED_NORMALIZED] = df[BEDROOMS].apply(lambda s: s if s<=3 else 3)
    def norm_bath(s):
        s=round(s)
        if s==0:
            return 1
        if s>=2:
            return 2
        return s

    df[BATH_NORMALIZED]=df[BATHROOMS].apply(norm_bath)



def process_nei123(train_df, test_df):
    df = pd.concat([train_df, test_df])
    normalize_bed_bath(df)
    sz= float(len(df))
    # neis_cols = [NEI_1, NEI_2, NEI_3]
    new_cols=[]
    for col in [NEI_1, NEI_2]:
        new_col = 'freq_of_{}'.format(col)
        df[new_col] = df.groupby(col)[PRICE].transform('count')
        df[new_col] = df[new_col]/sz
        new_cols.append(new_col)

    beds_vals =[0,1,2,3]
    for col in [NEI_1, NEI_2, NEI_3]:
        for bed in beds_vals:
            new_col = 'freq_of_{}, with bed={}'.format(col, bed)
            df[new_col] = df.groupby([col, BED_NORMALIZED])[PRICE].transform('count')
            df[new_col] = df[new_col]/sz
            new_cols.append(new_col)

    for col in [NEI_1, NEI_2]:
        new_col = 'median_ratio_of_{}'.format(col)
        df['tmp'] = df.groupby([col, BEDROOMS])[PRICE].transform('median')
        df[new_col] = df[PRICE]-df['tmp']
        df[new_col] = df[new_col]/df['tmp']
        new_cols.append(new_col)


    # for col in [NEI_1, NEI_2, NEI_3]:
    #     vals = set(df[col])
    #     if None in vals:
    #         vals.remove(None)
    #     df = pd.get_dummies(df, columns=[col])
    #     dummies= get_dummy_cols(col, vals)
    #     new_cols+=dummies

    for d in [train_df, test_df]:
        for col in new_cols:
            d[col]=df.loc[d.index, col]

    return train_df, test_df, new_cols






