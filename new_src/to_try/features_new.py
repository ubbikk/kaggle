from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd


def process_features_new(X_train, X_test):
    fmt = lambda feat: [s.replace("\u00a0", "").strip().lower().replace(" ", "_") for s in feat]  # format features
    X_train["features"] = X_train["features"].apply(fmt)
    X_test["features"] = X_test["features"].apply(fmt)
    features = [f for f_list in list(X_train["features"]) + list(X_test["features"]) for f in f_list]
    ps = pd.Series(features)
    grouped = ps.groupby(ps).agg(len)
    features = grouped[grouped >= 10].index.sort_values().values    # limit to features with >=10 observations
    mlb = MultiLabelBinarizer().fit([features])
    columns = ['feature_' + s for s in mlb.classes_]
    flt = lambda l: [i for i in l if i in mlb.classes_]     # filter out features not present in MultiLabelBinarizer
    X_train = X_train.join(pd.DataFrame(data=mlb.transform(X_train["features"].apply(flt)), columns=columns, index=X_train.index))
    X_test = X_test.join(pd.DataFrame(data=mlb.transform(X_test["features"].apply(flt)), columns=columns, index=X_test.index))

    return X_train, X_test, columns