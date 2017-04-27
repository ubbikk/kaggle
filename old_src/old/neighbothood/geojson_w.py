import json
import pandas as pd
from collections import OrderedDict

import geojson

# fp='/home/dpetrovskyi/Desktop/NYC Address Points.geojson'
# bl = geojson.load(open(fp))
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
BOROCODE = 'borocode'


def convert_to_data_frame(raw_geo):
    bl = raw_geo['features']
    # bl = raw_geo
    lat = []  # 2
    long = []  # 1
    address_id = []
    borocode = []
    st_name = []
    zip_code = []
    side_of_st = []

    for f in bl:
        lat.append(f.get('geometry')["coordinates"][1])
        long.append(f.get('geometry')["coordinates"][0])

        properties = f.get('properties')
        if 'address_id' in properties:
            address_id.append(properties['address_id'])
        else:
            address_id.append(None)

        if 'borocode' in properties:
            borocode.append(properties['borocode'])
        else:
            borocode.append(None)

        if 'st_name' in properties:
            st_name.append(properties['st_name'])
        else:
            st_name.append(None)

        if 'zipcode' in properties:
            zip_code.append(properties['zipcode'])
        else:
            zip_code.append(None)

        if 'side_of_st' in properties:
            side_of_st.append(properties['side_of_st'])
        else:
            side_of_st.append(None)

    return pd.DataFrame(OrderedDict([
        ('latitude', lat),
        ('longitude', long),
        ('borocode', borocode),
        ('st_name', st_name),
        ('zip_code', zip_code),
        ('address_id', address_id),
        ('side_of_st', side_of_st),
    ]))

# def parse_tuple(t):
#     t=t[1:len(t)-1]
#     t=t.replace(',', '')
#     t=t.split()
#     return (float(t[0]), float(t[1]))
#
# def load_fp(fp):
#     bl=json.load(open(fp))
#     bl= {parse_tuple(k): v for k, v in bl.iteritems()}
#     bl= [(k,v[0]) for k,v in bl.iteritems()]
#     return pd.DataFrame({
#         LONGITUDE:[x[0][0] for x in bl],
#         LATITUDE:[x[0][1] for x in bl],
#         BOROCODE:[x[1] for x in bl]
#     })
