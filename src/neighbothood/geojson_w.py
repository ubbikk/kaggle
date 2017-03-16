import geojson

# fp='/home/dpetrovskyi/Desktop/NYC Address Points.geojson'
# bl = geojson.load(open(fp))

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
