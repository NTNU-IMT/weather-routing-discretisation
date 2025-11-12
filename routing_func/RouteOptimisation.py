# Copyright (c) 2025, NTNU
# Author: Aurore Wendling <aurore.wendling@ntnu.no>
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

import math
from dataclasses import dataclass
import xarray as xr
import numpy as np
import time
import searoute as sr
import warnings

from .tools import *
import json
from .rust_tools import *

@dataclass
class RouteOptimisation():

    departure_time: datetime
    latest_arrival_time: datetime
    waypoints: list
    max_total_energy: float
    stop_time_hours: int
    polar: xr.Dataset

    forward_res: float
    side_res: float
    time_res:int
    n_time_steps: int
    n_side_points: int
    n_adj_points: int
    n_stages = []

    gcr = []
    graph = np.empty((1,)) 
    space_grid: np.ndarray
    node_reachability_matrix: np.ndarray 
    stop_time_matrix: np.ndarray 

    weather_twa: np.ndarray 
    weather_tws: np.ndarray 
    weather_wa: np.ndarray
    weather_hs: np.ndarray

    wind_data_file: str
    wave_data_file: str
    

    courant_number: float


    def __init__(self,
                 departure_time,
                 waypoints,
                 stop_time_hours,
                 polar,
                 wind_data_file,
                 wave_data_file,
                 max_total_energy,
                 latest_arrival_time,
                 forward_res,
                 side_res,
                 n_side_points,
                 n_adj_points,
                 time_res):
        self.departure_time = departure_time
        self.waypoints = waypoints
        self.stop_time_hours = stop_time_hours
        self.polar = polar
        self.wind_data_file = wind_data_file
        self.wave_data_file = wave_data_file
        self.max_total_energy = max_total_energy
        self.latest_arrival_time = latest_arrival_time
        self.forward_res = forward_res
        self.side_res = side_res
        self.n_side_points = n_side_points
        self.n_adj_points = n_adj_points
        self.time_res = time_res
        max_voyage_time = self.latest_arrival_time - self.departure_time
        self.n_time_steps = int((max_voyage_time.days * 24 + max_voyage_time.seconds // 3600) // time_res)
        max_speed = np.max(polar.stw_ms.data)
        self.courant_number = (max_speed * self.time_res * 3600) / (self.forward_res * math.cos((self.n_side_points * self.side_res / 6378e3)))

    @classmethod
    def from_dict(cls, setup_dict:dict):
        return cls(
            departure_time=datetime.strptime(setup_dict['departure time'], '%Y-%m-%d %H:%M'),
            waypoints=setup_dict['waypoints'],
            stop_time_hours=setup_dict['stop time at intermediate points, hours'],
            polar=xr.open_dataset(setup_dict['polar']),
            wind_data_file=setup_dict['wind file'],
            wave_data_file=setup_dict['wave file'],
            max_total_energy=eval(setup_dict['maximum total energy kW']),
            latest_arrival_time=datetime.strptime(setup_dict['latest arrival time'], '%Y-%m-%d %H:%M'),
            forward_res=setup_dict['forward resolution km'] * 1000,
            side_res=setup_dict['side resolution km'] * 1000,
            n_side_points=setup_dict['number of side points'],
            n_adj_points=setup_dict['number of adjacent points'],
            time_res=setup_dict['time discretization hours']
        )

    @classmethod
    def from_setup_file(cls, setup_file:str):
        # Open setup file
        with open(setup_file, 'r') as openfile:
            input_parameters = json.load(openfile)
        # Open route file
        with open(input_parameters['waypoints'], 'r') as openfile:
            wp = json.load(openfile)
        # Initialize instance
        return cls(
            departure_time=datetime.strptime(input_parameters['departure time'], '%Y-%m-%d %H:%M'),
            waypoints=[(math.radians(v['lon']), math.radians(v['lat'])) for v in wp],
            stop_time_hours=input_parameters['stop time at intermediate points, hours'],
            polar=xr.open_dataset(input_parameters['polar']),
            wind_data_file=input_parameters['wind file'],
            wave_data_file=input_parameters['wave file'],
            max_total_energy=eval(input_parameters['maximum total energy kW']),
            latest_arrival_time=datetime.strptime(input_parameters['latest arrival time'], '%Y-%m-%d %H:%M'),
            forward_res=input_parameters['forward resolution km'] * 1000,
            side_res=input_parameters['side resolution km'] * 1000,
            n_side_points=input_parameters['number of side points'],
            n_adj_points=input_parameters['number of adjacent points'],
            time_res=input_parameters['time discretization hours']
        )

    # Discretise and create the adjacency graph
    def discretise(self, routing_mode='searoute', skip_courant_above_05=False, autocorrect_courant=False):
        if skip_courant_above_05 and self.courant_number >= 0.5:
            warnings.warn('Skipped: Courant number is too large')
        else:
            if self.courant_number >= 0.5:
                warnings.warn('Courant number is higher than 0.5 !!! ')
                if autocorrect_courant:
                    time_res = 0.45 * (self.forward_res * math.cos((self.n_side_points * self.side_res / 6378e3))) / (np.max(self.polar.stw_ms.data) * 3600)
                    max_voyage_time = self.latest_arrival_time - self.departure_time
                    self.time_res = (1/(2**(round(math.log(1/(time_res), 2))))) / 2
                    self.n_time_steps = int((max_voyage_time.days * 24 + max_voyage_time.seconds // 3600) // self.time_res)
                    print('Time resolution was set to ' + str(self.time_res) + 'h to correct Courant number.')
            match routing_mode:
                case 'searoute':
                    path_finding_function = self.find_shortest_path_with_searoute
                case 'great circle':
                    path_finding_function = self.find_great_circle_route
            self.find_shortest_route(path_finding_function)
            self.make_space_grid()
            self.make_graph()
            self.make_stop_time_matrix()
            self.make_reachability_matrix()

    # Prepare weather data
    def read_weather(self, include_waves=True):
        if not self.graph.any():
            print('Cannot interpolate weather because graph is empty')
        else:
            self.interp_wind(self.wind_data_file)
            if include_waves:
                self.interp_waves(self.wave_data_file)
            else:
                self.weather_wa = np.full(shape=self.node_reachability_matrix.shape, fill_value=EPSILON)
                self.weather_hs = np.full(shape=self.node_reachability_matrix.shape, fill_value=EPSILON)

    # Find the pareto-optimal set of routes        
    def find_routes(self):
        if not self.graph.any():
            print('Invalid graph')
            return [], [], [], [], []
        else:
            rust_input = rustify_input(
                self.graph,
                self.stop_time_matrix,
                self.n_time_steps,
                self.time_res,
                self.max_total_energy,
                self.space_grid,
                self.node_reachability_matrix,
                self.weather_twa,
                self.weather_tws,
                self.weather_wa,
                self.weather_hs,
                self.polar
            )
            finalres, lng, lat, index2mode = solve_with_rust(rust_input)
            costs, durations, places, times, modes = pythonify_output(finalres, lng, lat, index2mode)
            if costs.any():
                return remove_dominated_solutions(costs, durations, places, times, modes)
            else:
                return [], [], [], [], []

    # Interpolate a weather dataset to the grid
    def interp_weather(self, file):

        # Make the interpolation grid in a format xarray likes:
        longitudes = xr.DataArray(
            data=[math.degrees(self.space_grid[i, j][0])
                  for i in range(self.space_grid.shape[0])
                  for j in range(self.space_grid.shape[1])
                  if self.space_grid[i, j]],
            dims="points")
        latitudes = xr.DataArray(
            data=[math.degrees(self.space_grid[i, j][1])
                  for i in range(self.space_grid.shape[0])
                  for j in range(self.space_grid.shape[1])
                  if self.space_grid[i, j]],
            dims="points")

        dataset = xr.open_dataset(file)

        #To format TIGGE datasets:
        if 'time' in dataset.dims:
            dataset = dataset.rename({'time':'valid_time'})
            longitudes = xr.DataArray( #TIGGE datasets have longitudes from 0 to 360, while ERA5 datasets have longitudes from -180 to 180
                data=[math.degrees(self.space_grid[i, j][0]) % 360
                      for i in range(self.space_grid.shape[0])
                      for j in range(self.space_grid.shape[1])
                      if self.space_grid[i, j]],
                dims="points")

        # Reduce dataset size to data for the area and voyage time:
        dataset_end_time = datetime.utcfromtimestamp((dataset.valid_time.data[-1] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
        if dataset_end_time < self.latest_arrival_time:
            warnings.warn('The weather dataset has no data for maximum voyage time !!! ')
            #TODO:switch to forecast mode
        dataset = dataset.sel(
            valid_time=slice(self.departure_time, min(self.latest_arrival_time, dataset_end_time)),
            latitude=slice(max(latitudes), min(latitudes)),
            longitude=slice(min(longitudes), max(longitudes))
        )

        # Interpolate dataset to grid:
        dataset = dataset.interp(latitude=latitudes,
                                 longitude=longitudes,
                                 method='linear').interpolate_na(dim="points", method='linear', fill_value='extrapolate')
                                       
        # Reindex dataset with desired time resolution:
        time_index = dataset.coords.variables.mapping['valid_time'].values
        delta = np.timedelta64(int(round(self.time_res*3600*1000)),'ms')
        new_time_index = time_index[0] + range(0, self.n_time_steps) * np.array([delta]*self.n_time_steps)
        dataset = dataset.reindex({"valid_time": new_time_index})
        dataset = dataset.interpolate_na(dim='valid_time', method='linear', fill_value='extrapolate') #interpolate to desired shape and then interpolate missing values
        weather_array_dict = dataset.to_dict()

        dataset.close()

        return weather_array_dict     

    # Interpolate wind data
    def interp_wind(self, file):
        self.weather_twa = np.empty(shape=self.node_reachability_matrix.shape) 
        self.weather_tws = np.empty(shape=self.node_reachability_matrix.shape)
        
        weather_array_dict = self.interp_weather(file)

        point = 0
        for i in range(self.space_grid.shape[0]):
            for j in range(self.space_grid.shape[1]):
                if self.space_grid[i, j]:
                    for k in range(self.n_time_steps):
                        u10 = weather_array_dict['data_vars']['u10']['data'][k][point]
                        v10 = weather_array_dict['data_vars']['v10']['data'][k][point]
                        self.weather_twa[i, j, k] = compute_twa(u10, v10)
                        self.weather_tws[i, j, k] = compute_tws(u10, v10)
                    point += 1

    # Interpolate wave data   
    def interp_waves(self, file):
        self.weather_wa = np.empty(shape=self.node_reachability_matrix.shape)
        self.weather_hs = np.empty(shape=self.node_reachability_matrix.shape)

        weather_array_dict = self.interp_weather(file)

        point = 0
        for i in range(self.space_grid.shape[0]):
            for j in range(self.space_grid.shape[1]):
                if self.space_grid[i, j]:
                    for k in range(self.n_time_steps):
                        self.weather_wa[i, j, k] = math.radians(weather_array_dict['data_vars']['mwd']['data'][k][point]) % math.pi
                        self.weather_hs[i, j, k] = weather_array_dict['data_vars']['swh']['data'][k][point]
                    point += 1

    # Finds the shortest route passing through the input waypoints
    def find_shortest_route(self, path_finding_function):
        self.gcr = [self.waypoints[0]]
        for i in range(len(self.waypoints) - 1):
            this_leg = path_finding_function(self.waypoints[i], self.waypoints[i + 1])
            self.gcr += this_leg[1::]
            self.n_stages += [len(this_leg) - 1]

    # Finds the shortest route between two points, using the searoute library
    def find_shortest_path_with_searoute(self, start_point, end_point):
        sr_output = sr.searoute(
            (math.degrees(start_point[0]),
             math.degrees(start_point[1])),
            (math.degrees(end_point[0]),
             math.degrees(end_point[1])),
             restrictions=['northwest']
        ).geometry.coordinates
        sr_route = [(math.radians(point[0]), math.radians(point[1])) for point in sr_output]

        min_forward_res = (np.max(self.polar.stw_ms.data) * self.time_res * 3600 / 0.5) / math.cos((self.n_side_points * self.side_res / 6378e3))

        leg = [sr_route[0]]
        for i in range(1, len(sr_route)):
            dist_to_next = compute_shortest_distance(leg[-1], sr_route[i])
            if dist_to_next >= 2 * min_forward_res:
                subdivisions = int(dist_to_next // self.forward_res)
                subdivision_dist = dist_to_next / (subdivisions + 1)
                for n in range(subdivisions):
                    bearing_to_next_point = compute_bearing(start_point=leg[-1],
                                                            end_point=sr_route[i])
                    next_point = find_point_at_bearing_and_distance(start_point=leg[-1],
                                                                    bearing=bearing_to_next_point,
                                                                    distance=subdivision_dist)
                    leg += [next_point]
                leg += [sr_route[i]]
        return leg

    # Finds the great circle route between two points
    def find_great_circle_route(self, start_point, end_point):
        leg = [start_point]
        n_stages = int(compute_shortest_distance(start_point, end_point) // self.forward_res + 1)
        for i in range(n_stages -1):
            bearing_to_next = compute_bearing(leg[-1], end_point)
            next_point = find_point_at_bearing_and_distance(start_point=leg[-1],
                                                            bearing=bearing_to_next,
                                                            distance=self.forward_res)
            leg += [next_point]
        leg += [end_point]
        return leg

    # Create a matrix to store stopping time at each waypoint
    def make_stop_time_matrix(self):
        self.stop_time_matrix = np.zeros(shape=self.space_grid.shape)
        for i in range(self.space_grid.shape[0]):
            if i in [sum(self.n_stages[:n]) - 1 for n in range(len(self.n_stages))]:
                for j in range(self.space_grid.shape[1]):
                    self.stop_time_matrix[i,j] = self.stop_time_hours

    # Create a matrix to indicate reachable nodes (x,y,t)
    def make_reachability_matrix(self):
        max_speed = np.max(self.polar.stw_ms.data)
        self.node_reachability_matrix = np.zeros(shape=self.space_grid.shape + (self.n_time_steps,))

        for i in range(self.node_reachability_matrix.shape[0]):
            for j in range(self.node_reachability_matrix.shape[1]):
                if self.graph[i, j]:
                    min_time_from_start = custom_round(
                        compute_shortest_distance(self.space_grid[i, j], self.gcr[0]) / max_speed / 3600 / self.time_res)
                    min_time_to_destination = custom_round(
                        compute_shortest_distance(self.space_grid[i, j], self.gcr[-1]) / max_speed / 3600 / self.time_res)
                    for k in range(self.node_reachability_matrix.shape[2]):
                        if k - min_time_from_start >= 0 and k + min_time_to_destination < self.n_time_steps:
                            self.node_reachability_matrix[i,j,k] = 1

        self.graph[-1, self.n_side_points] = [] #remove dummy adjacent points to graph's end point

    # Create 2D grid around great circle route, excluding points on land
    def make_space_grid(self):
        m = Basemap(resolution='c')  # used by the library that checks if arcs cross land
        self.space_grid = np.full(shape=[len(self.gcr), 2 * self.n_side_points + 1], fill_value=None)
        self.space_grid[0, self.n_side_points] = self.gcr[0]
        for i in range(1, self.space_grid.shape[0]-1):
            self.space_grid[i, self.n_side_points] = self.gcr[i]
            for j in range(1, self.n_side_points): #create new points perpendicular to the route
                for sign in [-1, 1]: #in both directions
                    if i == self.space_grid.shape[0] - 2:
                        bearing_to_next_point = (sign * math.pi/2 + compute_bearing(start_point=self.gcr[i],
                                                                                    end_point=self.gcr[i + 1])
                                                 + sign * math.pi/2 + compute_bearing(start_point=self.gcr[i-1],
                                                                                      end_point=self.gcr[i])) / 2
                    else:
                        bearing_to_next_point = (sign * math.pi/2 + compute_bearing(start_point=self.gcr[i],
                                                                                    end_point=self.gcr[i + 1])
                                                 + sign * math.pi/2 + compute_bearing(start_point=self.gcr[i-1],
                                                                                      end_point=self.gcr[i])
                                                 + sign * math.pi/2 + compute_bearing(start_point=self.gcr[i+1],
                                                                                      end_point=self.gcr[i + 2])) / 3
                    new_point = find_point_at_bearing_and_distance(start_point=self.gcr[i],
                                                                   bearing=bearing_to_next_point,
                                                                   distance=j * self.side_res)
                    if m.is_sea(math.degrees(new_point[0]), math.degrees(new_point[1])): # check that the new points are not on land
                        self.space_grid[i, self.n_side_points + sign * j] = new_point
            self.space_grid[-1, self.n_side_points] = self.gcr[-1]

        # Remove crossing lines
        max_speed = np.max(self.polar.stw_ms.data)
        min_arc_distance = max_speed * self.time_res * 3600 / 0.5
        for i1 in range(0, self.space_grid.shape[0]):
            for i2 in range(0, i1):
                theta_1 = compute_bearing(self.space_grid[i2, self.n_side_points],
                                          self.space_grid[i1, self.n_side_points])
                for j in range(self.space_grid.shape[1]):
                    if self.space_grid[i1, j] and self.space_grid[i2, j]:
                        theta_2 = compute_bearing(self.space_grid[i2, j], self.space_grid[i1, j])
                        d = compute_shortest_distance(self.space_grid[i2, j], self.space_grid[i1, j])
                        if (abs(theta_1 - theta_2) >= math.pi / 2 or d <= min_arc_distance) and j != self.n_side_points:
                            self.space_grid[i1, j] = None

    # Create adjacency graph from space grid, excluding arcs that cross land and dead ends
    def make_graph(self):
        m = Basemap(resolution='c')  # used by the library that checks if arcs cross land
        self.graph = np.full(shape=self.space_grid.shape, fill_value=None)
        self.graph[-1,self.n_side_points] = [(0,0)] ## Dummy

        # Check for land crossings
        for i in range(0,self.graph.shape[0]-1):
            self.graph[i, self.n_side_points] = [(i+1, self.n_side_points)]
            for j in range(self.graph.shape[1]):
                if self.space_grid[i,j]:
                    for c in range(-self.n_adj_points, self.n_adj_points + 1):
                        if 0 <= j + c < self.graph.shape[1]:
                            if self.space_grid[i + 1, j + c] and not arc_crosses_land(self.space_grid[i, j], self.space_grid[i + 1, j + c], m):
                                    if self.graph[i, j]:
                                        self.graph[i,j].append((i + 1, j + c))
                                    else:
                                        self.graph[i,j] = [(i + 1, j + c)]

        # Remove dead ends
        while not all([any(((i,j) in self.graph[i-1,c] if self.graph[i-1,c] else []) for c in range(self.graph.shape[1]))
                       for i in range(1,self.graph.shape[0]) for j in range(self.graph.shape[1])
                       if self.graph[i,j]]):
            for i in range(1, self.graph.shape[0]):
                for j in range(self.graph.shape[1]):
                    if not any(((i,j) in self.graph[i-1,c] if self.graph[i-1,c] else []) for c in range(self.graph.shape[1])):
                        self.graph[i,j] = None
        while not all([self.graph[l[c]] for l in (self.graph[i,j] for i in range(0,self.graph.shape[0]-1) for j in range(self.graph.shape[1]) if self.graph[i,j]) for c in range(len(l))]):
            for i in range(1, self.graph.shape[0]-1):
                for j in range(self.graph.shape[1]):
                    if self.graph[i,j]:
                        for p in self.graph[i,j]:
                            if not self.graph[p]:
                                self.graph[i, j].remove(p)

        # Empty the graph if the departure point is unreachable:
        if not self.graph[0, self.n_side_points]:
            self.graph.fill([])
        if not self.graph.any():
            print('Graph is empty')

        # Update space grid to remove unreachable points:
        for i in range(self.space_grid.shape[0]):
            for j in range(self.space_grid.shape[1]):
                if not self.graph[i, j]:
                    self.space_grid[i, j] = None

    def count_spatial_edges(self):
        n_edges = 0
        for i in range(0,self.graph.shape[0]-1):
            for j in range(self.graph.shape[1]):
                if self.space_grid[i,j]:
                    n_edges += len(self.space_grid[i,j])
        return n_edges
