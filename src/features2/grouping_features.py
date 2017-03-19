from collections import OrderedDict

F_COL = u'features'
COL = 'normalized_features'


def normalize_df(df):
    df[COL] = df[F_COL].apply(lambda l: [x.lower() for x in l])


def lambda_in(in_arr):
    def is_in(l):
        for f in l:
            for t in in_arr:
                if t in f:
                    return 1

        return 0

    return is_in


def lambda_equal(val):
    def is_equal(l):
        for f in l:
            if f.strip() == val:
                return 1

        return 0

    return is_equal


def lambda_two_arr(arr1, arr2):
    def is_in(l):
        for f in l:
            for x in arr1:
                for y in arr2:
                    if x in f and y in f:
                        return 1
        return 0

    return is_in


GROUPING_MAP = OrderedDict([
    ('elevator', {'vals': ['elevator']}),
    ('hardwood floors', {'vals': ['hardwood']}),
    ('cats allowed', {'vals': ['cats']}),
    ('dogs allowed', {'vals': ['dogs']}),
    ('doorman', {'vals': ['doorman', 'concierge']}),
    ('dishwasher', {'vals': ['dishwasher']}),
    ('laundry in building', {'vals': ['laundry']}),
    ('no fee', {'vals': ['no fee', 'no broker fee', 'no realtor fee']}),
    ('reduced fee', {'vals': ['reduced fee']}),
    ('fitness center', {'vals': ['fitness']}),
    ('pre-war', {'vals': ['pre-war', 'prewar']}),
    ('roof deck', {'vals': ['roof']}),
    ('outdoor space', {'vals': ['outdoor space', 'outdoor-space', 'outdoor areas', 'outdoor entertainment']}),
    ('common outdoor space', {'vals': ['common outdoor', 'publicoutdoor', 'public-outdoor', 'common-outdoor']}),
    ('private outdoor space', {'vals': ['private outdoor', 'private-outdoor', 'privateoutdoor']}),
    ('dining room', {'vals': ['dining']}),
    ('high speed internet', {'vals': ['internet']}),
    ('balcony', {'vals': ['balcony']}),
    ('swimming pool', {'vals': ['swimming', 'pool']}),
    ('new construction', {'vals': ['new construction']}),
    ('terrace', {'vals': ['terrace']}),
    ('exclusive', {'vals': ['exclusive']}),
    ('loft', {'vals': ['loft']}),
    ('garden/patio', {'vals': ['garden']}),
    ('wheelchair access', {'vals': ['wheelchair']}),
    ('fireplace', {'vals': ['fireplace']}),
    ('simplex', {'vals': ['simplex']}),
    ('lowrise', {'vals': ['lowrise', 'low-rise']}),
    ('garage', {'vals': ['garage']}),
    ('reduced fee', {'vals': ['reduced fee', 'reduced-fee', 'reducedfee']}),
    ('furnished', {'vals': ['furnished']}),
    ('multi-level', {'vals': ['multi-level', 'multi level', 'multilevel']}),
    ('high ceilings', {'vals': ['high ceilings', 'highceilings', 'high-ceilings']}),
    ('parking space', {'vals': ['parking']}),
    ('terrace', {'vals': ['terrace']}),
    ('live in super', {'vals': ['super'], 'vals2': ['live', 'site']}),
    ('renovated', {'vals': ['renovated']}),
    ('green building', {'vals': ['green building']}),
    ('storage', {'vals': ['storage']}),
    ('washer', {'vals': ['washer']}),
    ('stainless steel appliances', {'vals': ['stainless']})
])
