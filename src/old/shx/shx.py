import fiona
import shapely
import shapely.geometry
import shapefile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

shp = '/home/dpetrovskyi/Desktop/gis/ZillowNeighborhoods-NY/ZillowNeighborhoods-NY.shp'
dbf = '/home/dpetrovskyi/Desktop/gis/ZillowNeighborhoods-NY/ZillowNeighborhoods-NY.dbf'
shx = '/home/dpetrovskyi/Desktop/gis/ZillowNeighborhoods-NY/ZillowNeighborhoods-NY.shx'
NEIGHBOURHOOD = "neighbourhood"
LATITUDE = 'latitude'
LONGITUDE = 'longitude'


def read_records():
    shape_reader = shapefile.Reader(shp=open(shp, 'rb'), dbf=open(dbf, 'rb'), shx=open(shx, 'rb'))
    shapeRecords = shape_reader.shapeRecords()
    return [x.record for x in shapeRecords]

RECORDS = read_records()

def process_df(df):
    with fiona.open(shp) as fiona_collection:
        counter=-1
        df[NEIGHBOURHOOD] = [[] for x in range(len(df))]
        for shapefile_record in fiona_collection:
            counter+=1
            print counter
            shape = shapely.geometry.asShape( shapefile_record['geometry'] )
            town = RECORDS[counter][2]
            if town != 'New York':
                continue
            def append_nei(s):
                point = shapely.geometry.Point(s[LONGITUDE], s[LATITUDE])
                if shape.contains(point):
                    s[NEIGHBOURHOOD].append(RECORDS[counter])

            df.apply(append_nei, axis=1)

