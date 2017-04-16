import json

seeds_fp = '../seeds.json'

splits_big_fp='../splits_big.json'
splits_small_fp='../splits_small.json'
SPLITS_BIG=json.load(open(splits_big_fp))
SPLITS_SMALL=json.load(open(splits_small_fp))