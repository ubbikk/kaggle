from __future__ import print_function
import pandas as pd

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


def normalize_bed_bath(df):
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

def process_median_price(train_df, test_df):
    df = pd.concat([train_df, test_df])
    col = 'bed_bath_median'
    df[col] = df.groupby([BED_NORMALIZED, BATH_NORMALIZED])[PRICE].transform('median')
    train_df[col]=df.loc[train_df.index, col]
    test_df[col]=df.loc[test_df.index, col]

    return train_df, test_df

def process_nei12_median(train_df, test_df):
    min_num=20
    df = pd.concat([train_df, test_df])

    groupby = df.groupby([BED_NORMALIZED, BATH_NORMALIZED, NEI_2, NEI_1])
    df['count_by_nei2_nei1']= groupby[PRICE].transform('count')
    df['median_by_nei2_nei1']= groupby[PRICE].transform('median')

    groupby = df.groupby([BED_NORMALIZED, BATH_NORMALIZED, NEI_2])
    df['count_by_nei2']= groupby[PRICE].transform('count')
    df['median_by_nei2']= groupby[PRICE].transform('median')

    groupby = df.groupby([BED_NORMALIZED, BATH_NORMALIZED])
    df['count_by_bed_bath']= groupby[PRICE].transform('count')
    df['median_by_bed_bath']= groupby[PRICE].transform('median')

    small = df['count_by_nei2'] < min_num
    df.loc[small, 'median_by_nei2'] = df.loc[small, 'median_by_bed_bath']

    small = df['count_by_nei2_nei1'] < min_num
    df.loc[small, 'median_by_nei2_nei1'] = df.loc[small, 'median_by_nei2']

    diff_col='median_diff_by_nei2_nei1'
    ratio_col='median_ratio_by_nei2_nei1'

    bed_bath_diff_col='median_diff_by_bed_bath'
    bed_bath_ratio_col='median_ratio_by_bad_bath'

    train_df[diff_col]=df.loc[train_df.index, 'median_by_nei2_nei1']
    train_df[diff_col]=(train_df[PRICE]-train_df[diff_col])
    train_df[ratio_col]=train_df[diff_col]/train_df[PRICE]

    train_df[bed_bath_diff_col]=df.loc[train_df.index, 'median_by_bed_bath']
    train_df[bed_bath_diff_col]=(train_df[PRICE]-train_df[bed_bath_diff_col])
    train_df[bed_bath_ratio_col]=train_df[bed_bath_diff_col]/train_df[PRICE]

    test_df[diff_col]=df.loc[test_df.index, 'median_by_nei2_nei1']
    test_df[diff_col]=(test_df[PRICE]-test_df[diff_col])
    test_df[ratio_col]=test_df[diff_col]/test_df[PRICE]

    test_df[bed_bath_diff_col]=df.loc[test_df.index, 'median_by_bed_bath']
    test_df[bed_bath_diff_col]=(test_df[PRICE]-test_df[bed_bath_diff_col])
    test_df[bed_bath_ratio_col]=test_df[bed_bath_diff_col]/test_df[PRICE]

    return train_df, test_df, [diff_col, ratio_col, bed_bath_diff_col,bed_bath_ratio_col]