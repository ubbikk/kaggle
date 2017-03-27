import pymongo as mng
from pymongo import MongoClient

def test1():
    client = MongoClient('10.20.0.144', 27017)
    db = client.renthop_results
    collection = db.test1
    # collection.insert_one(
    #     {
    #         'importance':[
    #             [5,3,5],
    #             [5,7,9]
    #         ],
    #         'results':[
    #             [1,2,3],
    #             [7,8,9]
    #         ]
    #     }
    # )

    return [x for x in collection.find()]

