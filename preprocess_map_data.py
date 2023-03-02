import os
import pickle
from pyrosm import OSM, get_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--score_type", help="Input type of score would like to preprocess/update ['business' or 'transporation']",type=str)
parser.add_argument("--update_pbf", help="Update map information file (Protobuf)",type=bool)
args = parser.parse_args()
score_type = args.score_type
if args.score_type not in ['business', 'transporation']:
    raise Exception("Undefine score type")

LOAD_FILE = './data/{}_filter_dict.pickle'.format(score_type)
SAVE_FILE = './data/{}-latest.csv'.format(score_type)

if os.path.isfile('./data/Bangkok.osm.pbf'):
    if args.update_pbf:
        fp = get_data("bangkok", directory="/data", update=True)
    else:
        fp = './data/Bangkok.osm.pbf'
else:
    fp = get_data("bangkok", directory="/data", update=True)
osm = OSM(fp)


with open(LOAD_FILE, 'rb') as f:
    my_filter, combine_dict = pickle.load(f)
    print('Finish load infos from: {}'.format(LOAD_FILE))

filter_key = list(my_filter.keys())
pois = osm.get_pois(custom_filter=my_filter)
merged_filter_col = pois[filter_key[0]].fillna('')
merged_filter_col = pois[filter_key[0]].fillna('')
[merged_filter_col := merged_filter_col+pois[filter_key[i]].fillna('') for i in range(1, len(filter_key))]
pois['poi_type'] = merged_filter_col
pois.to_csv(SAVE_FILE)
print('Finish save preprocess data to: {}'.format(SAVE_FILE))
