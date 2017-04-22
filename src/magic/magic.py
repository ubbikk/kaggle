import pandas as pd

# magic_file = '../../data/redhoop/listing_image_time.csv'
magic_file = '../data/redhoop/listing_image_time.csv'
LISTING_ID = 'listing_id'


def process_magic(train_df, test_df):
    image_date = pd.read_csv(magic_file)
    image_date.loc[80240,"time_stamp"] = 1478129766
    # image_date.loc[image_date['Listing_Id']==7119094, "time_stamp"] = 1478129766
    image_date["img_date"] = pd.to_datetime(image_date["time_stamp"], unit="s")
    image_date["img_days_passed"] = (image_date["img_date"].max() - image_date["img_date"]).astype(
        "timedelta64[D]").astype(int)
    image_date["img_date_month"] = image_date["img_date"].dt.month
    image_date["img_date_week"] = image_date["img_date"].dt.week
    image_date["img_date_day"] = image_date["img_date"].dt.day
    image_date["img_date_dayofweek"] = image_date["img_date"].dt.dayofweek
    image_date["img_date_dayofyear"] = image_date["img_date"].dt.dayofyear
    image_date["img_date_hour"] = image_date["img_date"].dt.hour
    image_date["img_date_minute"] = image_date["img_date"].dt.minute
    image_date["img_date_second"] = image_date["img_date"].dt.second
    image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(
        lambda x: 1 if x < 10 else 2 if x < 20 else 3)

    df = pd.concat([train_df, test_df])
    df = pd.merge(df, image_date, left_on=LISTING_ID, right_on='Listing_Id')
    new_cols = ["img_days_passed","img_date_month","img_date_week",
                "img_date_day","img_date_dayofweek","img_date_dayofyear",
                "img_date_hour", "img_date_monthBeginMidEnd",
                "img_date_minute", "img_date_second"]#+["img_date", "time_stamp"]

    for col in new_cols:
        train_df[col] = df.loc[train_df.index, col]
        test_df[col] = df.loc[test_df.index, col]

    return train_df, test_df, new_cols
