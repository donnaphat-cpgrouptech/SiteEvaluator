import json
import pickle
import geopandas
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from shapely import wkt
from shapely.geometry import Polygon
from AreaMap import boundingBox

def replace_or_combine_tag(df, keyword, replace_word):
    filter = np.where(df['poi_type'].str.contains(keyword, case=False, na=False), True, False)
    df.loc[filter, 'poi_type']=replace_word

def score_function(categoried_df):
    # Concat each point's dataframe to main dataframe
    df = pd.concat(categoried_df, axis=1).fillna(0)
    # prepare list for keep sum of rank score for each point
    # if we have 4 interested point max rank score = 4, min rank score = 1
    sum_rank = [0 for i in range(len(categoried_df))]

    description = {'Point {}'.format(i):[] for i in range(len(categoried_df))}
    desc_key = list(description.keys())

    for index, row in df.iterrows():
        row = [row[key] for key in desc_key]
        rank = rankdata(row, method='max')
        for i in range(len(desc_key)):
            description[desc_key[i]].append('This location have {} score rank: {}'.format(index, (len(categoried_df)-rank[i])+1))
        sum_rank+=rank
    
    # Compute score (Max score=100)
    score = (sum_rank/len(df.index))*(100/len(categoried_df))

    # last row is a overall score
    df.loc[len(df)] = score
    # get the all index and set last index to "Overall score".
    as_list = df.index.tolist()
    idx = as_list.index(len(df)-1)
    as_list[idx] = 'Overall score'
    df.index = as_list
    return df, pd.DataFrame(description, index=as_list[:-1])


def return_format(clean_result, score_df, rank_df):
    score_json = json.loads(score_df.to_json())
    rank_json = json.loads(rank_df.to_json())

    ret = {}
    for i in range(len(clean_result)):
        key = 'Point {}'.format(i)
        
        ret[key] = {
            'score' : score_json[key],
            'rank': rank_json[key],
            'properties': [item['properties'] for item in json.loads(clean_result[i].to_json())['features']] #Extract only real data, ignore auto-generate from pandas
        }
    return ret


def get_data(score_type):
    # Load preprocessed filter, combine dict
    with open('./data/{}_filter_dict.pickle'.format(score_type), 'rb') as f:
        my_filter, combine_dict = pickle.load(f)
    # Retrieve the preprocessed data and convert from pandas to geopandas
    bangkok_df = pd.read_csv('./data/{}-latest.csv'.format(score_type))
    bangkok_df['geometry'] = bangkok_df['geometry'].apply(wkt.loads)
    bangkok_gdf = geopandas.GeoDataFrame(bangkok_df, crs='epsg:4326')
    return bangkok_gdf, my_filter, combine_dict

def compute_score(data, score_type):
    interested_points = data["location"]
    if "inKm" in data.keys():
        inKm = data["inKm"]
    else:
        inKm = 1 # default

    bangkok_gdf, my_filter, combine_dict = get_data(score_type)

    # Create bounding box to scope area for place that we interested
    result = []
    for point in interested_points:
        bottom, left, top, right = list(boundingBox(point['lat'], point['long'], inKm))
        AreaPolygon = Polygon(((left, top), (right, top), (right, bottom), (left, bottom)))
        tmp = bangkok_gdf[bangkok_gdf['geometry'].covered_by(AreaPolygon)].reset_index(drop=True)
        tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
        result.append(tmp)

    # Change type of POIs together such as (bank, atm, exchange) => Finance
    for i in range(len(result)):
        for key, val in combine_dict.items():
            replace_or_combine_tag(result[i], key, val)

    # Drop row that (have name, tag, operator == N/A and duplicated row)
    clean_result = [
        result[i]
        .dropna(how='all', subset=['name', 'tags', 'operator'])
        .drop_duplicates(subset=['name', 'poi_type'])
        for i in range(len(result))
    ]

    # Group row that have same 'poi_type' value and count number
    categoried_result = [
        clean_result[i]
        .groupby('poi_type', group_keys=True)
        .apply(lambda x: int(len(x)))
        .to_frame(name="Point {}".format(i)) 
        for i in range(len(clean_result))
    ]

    # Compute score and return the result
    score_df, rank_df = score_function(categoried_result)    
    return return_format(clean_result, score_df, rank_df)


if __name__ == "__main__":
    data = {
        "location": [
            {
                "lat": 13.686216414042217, 
                "long": 100.51981659625747
            },
            {
                "lat": 13.671082321311063, 
                "long": 100.50920319063442
            }
        ],
        "restaurant_type": "อาหารไทย",
        "price_range": [
            {
                "price_from": "25",
                "price_to": "50"
            }
        ],
        "selling_dish": [
            "ผัดไทย",
            "ผัดกระเพาหมู",
            "ต้มข่าไก่",
            "ต้มมะระ",
            "ต้มปลาทู"
        ],
        "number_of_seats": "50",
        "available_service": [
            "accept_credit_card",
            "for_group",
            "for_children",
            "have_wifi",
            "have_parking"
        ],
        "operating_times": [
            {
                "opening_days": [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6"
                ],
                "opening_hour": "10:00",
                "closing_hour": "22:00"
            }
        ],
        "target_demographic": "เพื่อนฝูง",
        "shop_type": "restaurant"
    }
    result = {
        'transportation': compute_score(data, 'transportation'),
        'business': compute_score(data, 'business')
    }
    print(result)
    