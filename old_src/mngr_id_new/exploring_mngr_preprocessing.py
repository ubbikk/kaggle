import json
import os
from time import time

import seaborn as sns
import pandas as pd
from collections import OrderedDict

from matplotlib import pyplot
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.spatial import KDTree
import math



TARGET = u'interest_level'
TARGET_VALUES = ['low', 'medium', 'high']
MANAGER_ID = 'manager_id'
BUILDING_ID = 'building_id'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
PRICE = 'price'
BATHROOMS = 'bathrooms'
BEDROOMS = 'bedrooms'
DESCRIPTION = 'description'
DISPLAY_ADDRESS = 'display_address'
STREET_ADDRESS = 'street_address'
LISTING_ID = 'listing_id'
PRICE_PER_BEDROOM = 'price_per_bedroom'
F_COL=u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE='created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
NEI = 'neighbourhood'
BORO = 'boro'
NEI_1 = 'nei1'
NEI_2 = 'nei2'
NEI_3 = 'nei3'

# rent_file = '../with_geo/data/neis_from_renthop_lower.json'
rent_file = 'with_geo/data/neis_from_renthop_lower.json'

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



FEATURES = [u'bathrooms', u'bedrooms', u'building_id', u'created',
            u'description', u'display_address', u'features',
            u'latitude', u'listing_id', u'longitude', MANAGER_ID, u'photos',
            u'price', u'street_address']

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)

train_file = '../data/redhoop/train.json'
test_file = '../data/redhoop/test.json'

train_geo_file = '../data/redhoop/with_geo/train_geo.json'
test_geo_file = '../data/redhoop/with_geo/test_geo.json'


# train_file = '../../data/redhoop/train.json'
# test_file = '../../data/redhoop/test.json'
#
# train_geo_file = '../../data/redhoop/with_geo/train_geo.json'
# test_geo_file = '../../data/redhoop/with_geo/test_geo.json'

BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'


def normalize_bed_bath_good(df):
    df['bed_bath']=df[[BEDROOMS, BATHROOMS]].apply(lambda s: (s[BEDROOMS], s[BATHROOMS]), axis=1)
    def norm(s):
        bed=s[0]
        bath=s[1]
        if bed==0:
            if bath>=1.5:
                return [0,2.0]
        elif bed==1:
            if bath>=2.5:
                return [1,2.0]
        elif bed==2:
            if bath>=3.0:
                return [2,3.0]
        elif bed==3:
            if bath>=4.0:
                return [3,4.0]
        elif bed==4:
            if bath==0:
                return [4,1]
            elif bath>=4.5:
                return [4,4.5]
        elif bed>=5:
            if bath <=1.5:
                return [5,1.5]
            elif bath <=2.5:
                return [5,2.5]
            elif bath <=3.5:
                return [5,3]
            else:
                return [5,4]

        return [bed, bath]

    df['bed_bath']=df['bed_bath'].apply(norm)
    df[BED_NORMALIZED]=df['bed_bath'].apply(lambda s:s[0])
    df[BATH_NORMALIZED]=df['bed_bath'].apply(lambda s:s[1])

def load_df(file, geo_file):
    df = pd.read_json(file)
    geo = pd.read_json(geo_file)
    df[NEI]= geo[NEI]
    df['tmp']=df[NEI].apply(transform_geo_to_rent)
    df[NEI_1]=df['tmp'].apply(lambda s:None if s is None else s[0])
    df[NEI_2]=df['tmp'].apply(lambda s:None if s is None else s[1])
    df[NEI_3]=df['tmp'].apply(lambda s:None if s is None else s[2])
    normalize_bed_bath_good(df)
    return basic_preprocess(df)


def load_train():
    return load_df(train_file, train_geo_file)


def load_test():
    return load_df(test_file, test_geo_file)


def basic_preprocess(df):
    df['num_features'] = df[u'features'].apply(len)
    df['num_photos'] = df['photos'].apply(len)
    df['word_num_in_descr'] = df['description'].apply(lambda x: len(x.split(' ')))
    df["created"] = pd.to_datetime(df["created"])
    # df["created_year"] = df["created"].dt.year
    df[CREATED_MONTH] = df["created"].dt.month
    df[CREATED_DAY] = df["created"].dt.day
    df[CREATED_HOUR] = df["created"].dt.hour
    df[CREATED_MINUTE] = df["created"].dt.minute
    df[DAY_OF_WEEK] = df['created'].dt.dayofweek
    bc_price, tmp = boxcox(df['price'])
    df['bc_price'] = bc_price

    return df




BED_NORMALIZED = 'bed_norm'
BATH_NORMALIZED = 'bath_norm'

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





def hcc_encode(train_df, test_df, variable, binary_target, k=5, f=1, g=1, r_k=0.01, folds=5):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    prior_prob = train_df[binary_target].mean()
    hcc_name = "_".join(["hcc", variable, binary_target])

    skf = StratifiedKFold(folds)
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        grouped = big.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
        grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
        grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

        if hcc_name in small.columns:
            del small[hcc_name]
        small = pd.merge(small, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
        small.loc[small[hcc_name].isnull(), hcc_name] = prior_prob
        small[hcc_name] = small[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(small))
        train_df.loc[small.index, hcc_name] = small[hcc_name]

    grouped = train_df.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    test_df = pd.merge(test_df, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
    test_df.loc[test_df[hcc_name].isnull(), hcc_name] = prior_prob

    return train_df, test_df, hcc_name


def get_exp_lambda(k, f):
    def res(n):
        return 1 / (1 + math.exp(float(k - n) / f))

    return res


def process_mngr_categ_preprocessing(train_df, test_df):
    col = MANAGER_ID
    new_cols = []
    for df in [train_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df, col, binary_col)
        new_cols.append(new_col)

    return train_df, test_df, new_cols




def process_nei123(train_df, test_df):
    df = pd.concat([train_df, test_df])
    normalize_bed_bath(df)
    sz= float(len(df))
    # neis_cols = [NEI_1, NEI_2, NEI_3]
    new_cols=[]
    for col in [NEI_1, NEI_2]:
        new_col = 'freq_of_{}'.format(col)
        df[new_col] = df.groupby(col)[PRICE].transform('count')
        # df[new_col] = df[new_col]/sz
        new_cols.append(new_col)

    beds_vals =[0,1,2,3]
    for col in [NEI_1, NEI_2, NEI_3]:
        for bed in beds_vals:
            new_col = 'freq_of_{}, with bed={}'.format(col, bed)
            df[new_col] = df.groupby([col, BED_NORMALIZED])[PRICE].transform('count')
            # df[new_col] = df[new_col]/sz
            new_cols.append(new_col)

    for col in [NEI_1, NEI_2]:
        new_col = 'median_ratio_of_{}'.format(col)
        df['tmp'] = df.groupby([col, BEDROOMS])[PRICE].transform('median')
        df[new_col] = df[PRICE]-df['tmp']
        df[new_col] = df[new_col]/df['tmp']
        new_cols.append(new_col)
    for col in [NEI_1, NEI_2, NEI_3]:
        vals = set(df[col])
        if None in vals:
            vals.remove(None)
        df = pd.get_dummies(df, columns=[col])
        dummies= get_dummy_cols(col, vals)
        new_cols+=dummies

    for d in [train_df, test_df]:
        for col in new_cols:
            d[col]=df.loc[d.index, col]

    return train_df, test_df, new_cols

##################################################3
def process_mngr_target_ratios(train_df, test_df):
    return process_target_ratios(train_df, test_df, MANAGER_ID, 5)



def process_target_ratios(train_df, test_df, col, folds):
    skf = StratifiedKFold(folds)
    target_vals = ['high', 'medium', 'low']
    new_cols = {k: '{}_target_ratios_{}'.format(col, k) for k in target_vals}
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        calc_target_ratios(big, small, col, new_cols, train_df)

    calc_target_ratios(train_df.copy(), test_df.copy(),col, new_cols, update_df=test_df)

    return train_df, test_df, new_cols.values()


def calc_target_ratios(big, small, col, new_cols, update_df):
    target_vals = ['high', 'medium', 'low']
    dummies = {k:'target_cp_{}'.format(k) for k in target_vals}

    big['target_cp'] = big[TARGET].copy()
    big= pd.get_dummies(big, columns=['target_cp'])
    grouped = big.groupby(col).mean()
    small = pd.merge(small, grouped[dummies.values()], left_on=col, right_index=True)
    for t in target_vals:
        new_col = new_cols[t]
        update_df.loc[small.index, new_col] = small[dummies[t]]
##################################################3
def get_group_by_mngr_target_dummies(df):
    df = pd.get_dummies(df, columns=[TARGET])
    target_vals = ['high', 'medium', 'low']
    dummies = ['interest_level_{}'.format(k) for k in target_vals]
    return df.groupby(MANAGER_ID)[dummies].sum()

def get_target_means_by_mngr(df):
    df = pd.get_dummies(df, columns=[TARGET])
    target_vals = ['high', 'medium', 'low']
    dummies = ['interest_level_{}'.format(k) for k in target_vals]
    means= df.groupby(MANAGER_ID)[dummies].mean()
    means['count'] = df.groupby(MANAGER_ID)[MANAGER_ID].count()
    return means.sort_values(by=['count'], ascending=False)

def explore_bad_good_mngrs(df_to_explore):
    df = get_target_means_by_mngr(df_to_explore)
    df = df[df['count']>=50]
    return df



def explore_target(df):
    print 'high         {}'.format(len(df[df[TARGET]=='high'])/(1.0*len(df)))
    print 'medium       {}'.format(len(df[df[TARGET]=='medium'])/(1.0*len(df)))
    print 'low          {}'.format(len(df[df[TARGET]=='low'])/(1.0*len(df)))


def get_target_ratios_method(df):
    return df[[MANAGER_ID,'manager_id_target_ratios_low','manager_id_target_ratios_medium', 'manager_id_target_ratios_high']]
##################################################3
target_vals = ['high', 'medium', 'low']
train_df, test_df = load_train(), load_test()
train_df, test_df, new_cols = process_mngr_categ_preprocessing(train_df, test_df)
train_df, test_df, new_cols = process_nei123(train_df, test_df)
train_df, test_df, new_cols = process_mngr_target_ratios(train_df, test_df)

COLS_TO_SHOW_1=[ NEI_1, NEI_2,TARGET,PRICE, BEDROOMS, BATHROOMS,
                CREATED_HOUR, 'created',
                'median_ratio_of_nei1', 'median_ratio_of_nei2',
                'freq_of_nei2', 'freq_of_nei1',
                'num_features','num_photos','word_num_in_descr'
                ]

def show_mngr(mngr_id):
    return train_df[train_df[MANAGER_ID]==mngr_id][COLS_TO_SHOW_1]

df = explore_bad_good_mngrs(train_df)
high = df.sort_values(by='interest_level_high')
low = df.sort_values(by='interest_level_low')
#interesting_mngr_h='35f11f952ba96803a9d9e23e83e7f972'
#interesting_mngr_l='d1762ef0af965cfb5946ba0e209cc1c5'

