from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from time import time

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
F_COL = u'features'
CREATED_MONTH = "created_month"
CREATED_DAY = "created_day"
CREATED_MINUTE = 'created_minute'
CREATED_HOUR = 'created_hour'
DAY_OF_WEEK = 'dayOfWeek'
CREATED='created'
LABEL='lbl'


def hcc_encode(train_df, test_df, variable, binary_target,N,  k=5, f=1, g=1, r_k=0.01, folds=5):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    prior_prob = train_df[binary_target].mean()
    hcc_name = "_".join(["hcc", variable, binary_target])

    random_state = int(time())
    skf = StratifiedKFold(folds, random_state=random_state, shuffle=True)
    print random_state
    results=[]
    for _ in range(N):
        for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
            big = train_df.iloc[big_ind]
            small = train_df.iloc[small_ind]

            update_df = train_df[PRICE].to_frame('bl')
            results.append(update_df)

            grouped = big.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
            grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
            grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

            if hcc_name in small.columns:
                del small[hcc_name]
            small = pd.merge(small, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
            small.loc[small[hcc_name].isnull(), hcc_name] = prior_prob
            small[hcc_name] = small[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(small))
            update_df.loc[small.index, hcc_name] = small[hcc_name]

    train_df[hcc_name]=np.mean([x[hcc_name] for x in results])

    grouped = train_df.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    test_df = pd.merge(test_df, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
    test_df.loc[test_df[hcc_name].isnull(), hcc_name] = prior_prob

    return train_df, test_df, hcc_name

def process_mngr_categ_preprocessing(train_df, test_df):
    col = MANAGER_ID
    new_cols = []
    N=10
    for df in [train_df, test_df]:
        df['target_high'] = df[TARGET].apply(lambda s: 1 if s == 'high' else 0)
        df['target_medium'] = df[TARGET].apply(lambda s: 1 if s == 'medium' else 0)
    for binary_col in ['target_high', 'target_medium']:
        train_df, test_df, new_col = hcc_encode(train_df, test_df,  col, binary_col,N)
        new_cols.append(new_col)

    return train_df, test_df, new_cols