import os
import math
#import numpy as np
#import pandas as pd
from tqdm import tqdm
import multiprocessing
from itertools import repeat

#import folium
import osmnx as ox
#import igraph as ig
import networkx as nx
import geopandas as gpd
#import mapclassify as mc

#from pyrosm.data import sources
from pyrosm import OSM, get_data
#from sklearn.neighbors import BallTree

# degrees to radians
def deg2rad(degrees):
    return math.pi*degrees/180.0
# radians to degrees
def rad2deg(radians):
    return 180.0*radians/math.pi

# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]

# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )

# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84
def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    lat = deg2rad(latitudeInDegrees)
    lon = deg2rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    latMin = lat - halfSide/radius
    latMax = lat + halfSide/radius
    lonMin = lon - halfSide/pradius
    lonMax = lon + halfSide/pradius

    return (rad2deg(latMin), rad2deg(lonMin), rad2deg(latMax), rad2deg(lonMax))

class AreaMap():
  def __init__(self, interestedPoint:list, interestedDistance:int=1):
    self.interestedPoint = interestedPoint
    self.interestedDistance = interestedDistance
    if os.path.isfile('./Bangkok.osm.pbf'):
        self.fp = './Bangkok.osm.pbf'
    else:
        self.fp = get_data("bangkok", directory="/", update=True)
    self.create_map_data()

  def create_map_data(self):
    '''
    create osm object and map's node and graph
    '''
    self.osm = []
    self.G = []
    i=0
    for item in tqdm(self.interestedPoint):
      bbox = list(boundingBox(item['lat'], item['long'], self.interestedDistance))
      self.osm.append(OSM(self.fp, bounding_box=[bbox[1], bbox[0], bbox[3], bbox[2]]))
      # graph_type = {}
      # n_walk, e_walk = osm[i].get_network(nodes=True, network_type="walking")
      # G.append(osm[i].to_graph(n_walk, e_walk, graph_type="networkx"))
      # if is_plot_path:
      #   ox.plot_graph(G[i], ax=axs[i])
      # i+=1

  def shortest_distance(self, source, target, graph, point):
    '''
      Using for Finding shortest path and distance in Km
      input:
        - source: tuple in format (longtitude, latitude)
        - target: tuple in format (longtitude, latitude)
        - graph: Type networkx.classes.multidigraph.MultiDiGraph
    '''
    source_node = ox.nearest_nodes(graph, source[0], source[1])
    target_node = ox.nearest_nodes(graph, target[0], target[1])
    
    dist = nx.shortest_path_length(graph, source_node, target_node, weight="length")
    dist_in_KM = dist/1000
    return dist_in_KM


  '''
  Finding the business, station or etc based on input filter.
  '''
  def find_around_area(self, filter):
    filter_key  = list(filter.keys())
    self.all_pois = []

    with multiprocessing.Pool() as pool:
      # call the function for each item in parallel, get results as tasks complete
      for pois in pool.starmap(OSM.get_pois, zip(self.osm, repeat(filter))):
        merged_filter_col = pois[filter_key[0]].fillna('')
        merged_filter_col = pois[filter_key[0]].fillna('')
        [merged_filter_col := merged_filter_col+pois[filter_key[i]].fillna('') for i in range(1, len(filter_key))]
        pois['poi_type'] = merged_filter_col
        self.all_pois.append(pois)
    return self.all_pois
