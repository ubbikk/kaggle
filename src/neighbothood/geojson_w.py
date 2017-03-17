import json
import pandas as pd

import geojson

# fp='/home/dpetrovskyi/Desktop/NYC Address Points.geojson'
# bl = geojson.load(open(fp))
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
BOROCODE='borocode'

def get_coordinates_from_feature(f):
    return f.get('geometry')["coordinates"]

def get_borocode_from_feature(f):
    return f.get('properties')['borocode']

def get_map():
    fp='/home/dpetrovskyi/Desktop/NYC Address Points.geojson'
    bl = geojson.load(open(fp))
    res = {}
    for f in bl['features']:
        coord = get_coordinates_from_feature(f)
        coord = (coord[0], coord[1])
        borocode = get_borocode_from_feature(f)
        if coord in res:
            res[coord].append(borocode)
        else:
            res[coord]=[borocode]

    return res

def parse_tuple(t):
    t=t[1:len(t)-1]
    t=t.replace(',', '')
    t=t.split()
    return (float(t[0]), float(t[1]))

def load_fp(fp):
    bl=json.load(open(fp))
    bl= {parse_tuple(k): v for k, v in bl.iteritems()}
    bl= [(k,v[0]) for k,v in bl.iteritems()]
    return pd.DataFrame({
        LONGITUDE:[x[0][0] for x in bl],
         LATITUDE:[x[0][1] for x in bl],
        BOROCODE:[x[1] for x in bl]
    })
