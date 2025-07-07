# Copyright (c) 2025, NTNU
# Author: Aurore Wendling <aurore.wendling@ntnu.no>
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

import math
from datetime import datetime, timedelta
from .basemap_extended_py311 import Basemap
from .CONSTANTS import *
from numba import jit, njit
import numpy as np
import searoute as sr
import scipy

# Finds the shortest distance between two points, using the searoute library
def route_length_with_searoute(start_point, end_point):
    return sr.searoute(
        (math.degrees(start_point[0]),
            math.degrees(start_point[1])),
        (math.degrees(end_point[0]),
            math.degrees(end_point[1])),
            restrictions=['northwest']
    ).properties['length']

@njit
def compute_twa(u, v):
    #ref from ECMWF: https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
    #twa = 3 * math.pi / 2 + math.atan2(u, v)  # anticlockwise angle from N
    twa = (math.pi + math.atan2(u,v)) % (2*math.pi) #wind **comes** clockwise from S: 90 <= and => 270
    if twa==0:
        return EPSILON
    else:
        return twa

@njit
def compute_tws(u, v):
    tws = math.sqrt(u ** 2 + v ** 2)
    if tws==0:
        return EPSILON
    else:
        return tws

@njit
def compute_shortest_distance(start_point: tuple,
                              end_point: tuple
                              ) -> float:
    return EARTH_RADIUS_M * compute_shortest_angular_distance(start_point,end_point)

@njit
def compute_shortest_angular_distance(start_point: tuple,
                              end_point: tuple
                              ) -> float:
    shortest_distance = math.acos(math.sin(start_point[1]) * math.sin(end_point[1]) +
                                  math.cos(start_point[1]) * math.cos(end_point[1]) *
                                  math.cos(end_point[0] - start_point[0]))
    return shortest_distance

@njit
def compute_bearing(start_point: tuple,
                    end_point: tuple
                    ) -> float:
    if start_point[0] < 0:
        start_lon = 2 * math.pi + start_point[0]
    else:
        start_lon = start_point[0]
    if end_point[0] < 0:
        end_lon = 2 * math.pi + end_point[0]
    else:
        end_lon = end_point[0]
    dlon = end_lon - start_lon
    bearing = math.atan2(math.sin(dlon)*math.cos(end_point[1]), math.cos(start_point[1]) * math.sin(end_point[1]) - math.sin(start_point[1]) * math.cos(end_point[1]) * math.cos(dlon))
    return bearing #clockwise from N

@njit
def compute_input_angle(start_point: tuple,
                        end_point: tuple,
                        true_angle:float):
    return (true_angle - compute_bearing(start_point, end_point)) % (2 * math.pi)

def compute_input_matrix(start_point: tuple,
                 end_point: tuple,
                 wind_matrix) -> tuple:
        # works with 2D matrix (tws,twa)
        input_wind_matrix = np.zeros(wind_matrix.shape)
        for twa_idx in range(wind_matrix.shape[1]):
            iwa = compute_input_angle(start_point, end_point, wind_matrix[0, twa_idx])
            iwa_idx = custom_round(np.clip(iwa, 0, 5 * (wind_matrix.shape[1] - 1)))
            input_wind_matrix[:, iwa_idx] = input_wind_matrix[:, iwa_idx] + wind_matrix[:, twa_idx]
        return input_wind_matrix

@njit
def find_point_at_bearing_and_distance(start_point: tuple,
                                       bearing: float,
                                       distance: float
                                       ) -> tuple:
    angular_distance = distance / EARTH_RADIUS_M
    y = math.asin(math.sin(start_point[1]) * math.cos(angular_distance) + math.cos(start_point[1]) * math.cos(bearing) * math.sin(angular_distance))
    x = start_point[0] + math.atan2(math.sin(bearing)*math.sin(angular_distance)*math.cos(start_point[1]), math.cos(angular_distance) - math.sin(start_point[1])*math.sin(y))
    if x > math.pi:
        x = - math.pi + abs(x) % math.pi
    if x < -math.pi:
        x = math.pi - abs(x) % math.pi
    return x, y

#tests if an arc crosses land using Basemap
def arc_crosses_land(start_node:tuple,
                     end_node:tuple,
                     m: Basemap
                     ) -> bool:
    x1 = math.degrees(start_node[0])
    x2 = math.degrees(end_node[0])
    if x1 * x2 < 0 and abs(x1) > 90 and abs(x2) > 90: #antimeridian crossings
        x1 = x1 % 360
        x2 = x2 % 360
    return m.intersects_land(x1=x1, y1=math.degrees(start_node[1]),
                             x2=x2, y2=math.degrees(end_node[1]))

#tests if an arc is too long or crosses land
def is_adjacent(start_point: tuple,
                end_point: tuple,
                max_distance: float,
                m: Basemap
                ) -> bool:
    return compute_shortest_distance(start_point,end_point) <= max_distance and not arc_crosses_land(start_point, end_point,m)


#faster custom rounding function for positive numbers
@njit
def custom_round(x:float):
    #return int(x + 0.6) #round up
    #return int(x) #round down
    return int(x + 0.5) #round to closest

def make_gcr(start_point, end_point, n_stages):
    gcr = [start_point]
    stage_length = compute_shortest_distance(start_point=start_point,
                                      end_point=end_point) / n_stages
    for i in range(n_stages - 1):
        bearing_to_next_point = compute_bearing(start_point=gcr[-1],
                                                end_point=end_point)
        next_point = find_point_at_bearing_and_distance(start_point=gcr[-1],
                                                        bearing=bearing_to_next_point,
                                                        distance=stage_length)
        gcr.append(next_point)
    return gcr

def hyperarea(costs, durations, min_cost, min_duration, max_cost, max_duration):
    y = (costs - min_cost) / (max_cost - min_cost) 
    x = (durations - min_duration) / (max_duration - min_duration)
    return scipy.integrate.trapezoid(y, x=x) 