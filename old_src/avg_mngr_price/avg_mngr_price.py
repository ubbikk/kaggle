from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

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


def process_mngr_avg_median_price(train_df, test_df):
    df = pd.concat([train_df, test_df])
    bed_bath_median = 'bed_bath_median'
    df[bed_bath_median] = df.groupby([BED_NORMALIZED, BATH_NORMALIZED])[PRICE].transform('median')

    bed_bath_diff = 'bed_bath_diff'
    df[bed_bath_diff]=df[PRICE]-df[bed_bath_median]

    bed_bath_raio = 'bed_bath_ratio'
    df[bed_bath_raio]=df[bed_bath_diff]/df['bed_bath_median']

    df['gr_by_mngr_bed_bath_diff_median']=df.groupby(MANAGER_ID)[bed_bath_diff].transform('median')
    df['gr_by_mngr_bed_bath_diff_mean']=df.groupby(MANAGER_ID)[bed_bath_diff].transform('mean')

    df['gr_by_mngr_bed_bath_ratio_median']=df.groupby(MANAGER_ID)[bed_bath_raio].transform('median')
    df['gr_by_mngr_bed_bath_ratio_mean']=df.groupby(MANAGER_ID)[bed_bath_raio].transform('mean')

    new_cols= ['bed_bath_diff','bed_bath_ratio',
               'gr_by_mngr_bed_bath_diff_median','gr_by_mngr_bed_bath_diff_mean',
               'gr_by_mngr_bed_bath_ratio_median', 'gr_by_mngr_bed_bath_ratio_mean' ]

    for col in new_cols:
        train_df[col]=df.loc[train_df.index, col]
        test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df, new_cols