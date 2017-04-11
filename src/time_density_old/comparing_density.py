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
CREATED = "created"
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
created_raw='created_raw'

def perform(train_df, test_df):
    df = pd.concat([train_df, test_df])
    df = df[df[DAY_OF_WEEK]==6]
    # df = df[df[NEI_2]=='midtown manhattan']
    df=df[df[CREATED_HOUR]==7]
    # df = df[df[BEDROOMS]==1]
    df=df[df[CREATED_MONTH]==4]
    df = df[[CREATED,CREATED_DAY, BEDROOMS,DAY_OF_WEEK, NEI_1]]
