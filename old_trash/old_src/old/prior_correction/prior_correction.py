interest_levels = ['low', 'medium', 'high']

tau = {
    'low': 0.69195995,
    'medium': 0.23108864,
    'high': 0.07695141,
}

def correct(df):
    y = df[interest_levels].mean()
    a = [tau[k] / y[k]  for k in interest_levels]
    print a

    def f(p):
        for k in range(len(interest_levels)):
            p[k] *= a[k]
        return p / p.sum()

    df_correct = df.copy()
    df_correct[interest_levels] = df_correct[interest_levels].apply(f, axis=1)

    y = df_correct[interest_levels].mean()
    a = [tau[k] / y[k]  for k in interest_levels]
    print a

    return df_correct