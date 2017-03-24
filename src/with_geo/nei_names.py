import json
def get_rent_nei_flat_fp(fp):
    rr = json.load(open(fp))
    res=[]
    for m in rr.values():
        for v in m.values():
            res+=v

    return res

