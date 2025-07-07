# Copyright (c) 2025, NTNU
# Authors: Aurore Wendling <aurore.wendling@ntnu.no>, Alberto Tamburini <albetam@dtu.dk>
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

import numpy as np
import ctypes
import time
import json
import xarray as xr
import math


####################################################
# Structure to hold results
####################################################
class CRoutes(ctypes.Structure):
    _fields_ = [
        ('costs', ctypes.POINTER(ctypes.c_double)),
        ('durations', ctypes.POINTER(ctypes.c_double)),
        ('places', ctypes.POINTER(ctypes.c_size_t)),
        ('times', ctypes.POINTER(ctypes.c_size_t)),
        ('modes', ctypes.POINTER(ctypes.c_size_t)),
        ('solutions_sizes', ctypes.POINTER(ctypes.c_size_t)),
        ('num_solutions', ctypes.c_size_t),
    ]

####################################################
# Flattening input and removing non-standard types (like None)
# I keep every iterable as a numpy type because if has a good interfacing to C/Rust
# Assuming no waves
####################################################
def rustify_input(
        graph,
        stop_time_matrix,
        n_time_steps,
        time_res,
        max_total_energy,
        space_grid,
        node_reachability_matrix,
        weather_twa,
        weather_tws,
        weather_wa,
        weather_hs,
        polar):
    ####################################################
    # Removing nans or empty vectors and flattening the graph
    # Moreover, compute the degree of each node of the graph
    ####################################################
    start_time = time.time()
    num_stages, stage_size = graph.shape
    start_point = (stage_size - 1) // 2
    end_point = (num_stages - 1) * stage_size + (stage_size - 1) // 2

    flat_graph = np.array([i * stage_size + j for node in graph.flatten() if node for (i, j) in node])
    flatgraph_lenths = np.array([len(node) if node else 0 for node in graph.flatten()])
    flatgraph_lenths_len = len(flatgraph_lenths)

    ####################################################
    # Only flattening the stop-time-matrix (optional)
    ####################################################
    start_time = time.time()
    flatten_stop_time_matrix = stop_time_matrix.flatten()

    ####################################################
    # Extracting the lng and lat coordinates
    # Substituting nones with infinity
    ####################################################
    start_time = time.time()
    lng = np.array([val[0] if val else np.inf for val in space_grid.flatten()])
    lat = np.array([val[1] if val else np.inf for val in space_grid.flatten()])

    ####################################################
    # Flattening the reachability matrix
    # Order is: [flattened_graph_index][time]
    ####################################################
    start_time = time.time()
    flatten_node_reachability_matrix = node_reachability_matrix.reshape(
        node_reachability_matrix.shape[0]*node_reachability_matrix.shape[1], node_reachability_matrix.shape[2]
    ).flatten().astype(dtype=bool)

    ####################################################
    # Flattening the weather grid and separating in its twa and tws components
    # Substituting nones with infinity
    # Order is: [flattened_graph_index][time]
    # THIS IS THE SLOWEST PART IN TRANSFORMATION DUE TO THE DICTIONARY
    # Worth to not build the dictionary since the beginning to avoid the overhead
    ####################################################

    flatten_twa = np.array([val if val else np.inf for val in weather_twa.reshape(
        weather_twa.shape[0] * weather_twa.shape[1], weather_twa.shape[2]
    ).flatten()])
    flatten_tws = np.array([val if val else np.inf for val in weather_tws.reshape(
        weather_tws.shape[0] * weather_tws.shape[1], weather_tws.shape[2]
    ).flatten()])
    flatten_wa = np.array([val if val else np.inf for val in weather_wa.reshape(
        weather_wa.shape[0] * weather_wa.shape[1], weather_wa.shape[2]
    ).flatten()])
    flatten_hs = np.array([val if val else np.inf for val in weather_hs.reshape(
        weather_hs.shape[0] * weather_hs.shape[1], weather_hs.shape[2]
    ).flatten()])

    ####################################################
    # Transforming modes into an ordered list and separating in its speed and power components
    ####################################################

    index2mode = list(polar.mode.data)
    modes_speed = polar.stw_ms.data
    modes_power = polar.brake_power_kw.data
    tws_coords = polar.tws_ms.data.astype(np.float64)
    twa_coords = np.array([round(math.radians(i),3) for i in polar.twa_deg.data])
    wa_coords = np.array([round(math.radians(i),3) for i in polar.wa_deg.data])
    hs_coords = polar.hs_m.data.astype(np.float64)
    modes_shape0, modes_shape1, modes_shape2, modes_shape3, modes_shape4 = modes_speed.shape

    return (
        start_point, end_point,
        flat_graph, flatgraph_lenths, flatgraph_lenths_len,
        flatten_stop_time_matrix,
        lng, lat,
        flatten_node_reachability_matrix,
        flatten_twa, flatten_tws, flatten_wa, flatten_hs,
        index2mode, modes_speed, modes_power, modes_shape0, modes_shape1, modes_shape2, modes_shape3, modes_shape4,
        twa_coords, tws_coords, wa_coords, hs_coords,
        n_time_steps, time_res, max_total_energy
    )

def solve_with_rust(rust_input):
    (
        start_point, end_point,
        flat_graph, flatgraph_lenths, flatgraph_lenths_len,
        flatten_stop_time_matrix,
        lng, lat,
        flatten_node_reachability_matrix,
        flatten_twa, flatten_tws, flatten_wa, flatten_hs,
        index2mode, modes_speed, modes_power, modes_shape0, modes_shape1, modes_shape2, modes_shape3, modes_shape4,
        twa_coords, tws_coords, wa_coords, hs_coords,
        n_time_steps, time_res, max_total_energy
    ) = rust_input
    rustlib = ctypes.cdll.LoadLibrary('./windroute/target/release/libwindroute.so')  # Load the library
    rustlib.graph_search_waves.restype = CRoutes
    finalres = rustlib.graph_search_waves(  # Run the algorithm
        ctypes.c_void_p(flat_graph.ctypes.data),  # Graph
        ctypes.c_void_p(flatten_stop_time_matrix.ctypes.data),  # Graph wait time
        ctypes.c_void_p(flatgraph_lenths.ctypes.data),  # Degree of each node of the graph
        ctypes.c_size_t(flatgraph_lenths_len),  # Number of nodes of the graph
        ctypes.c_void_p(lng.ctypes.data),  # Longitude coordinates
        ctypes.c_void_p(lat.ctypes.data),  # Latitude coordinates
        ctypes.c_void_p(flatten_node_reachability_matrix.ctypes.data),  # Time-space reachability matrix
        ctypes.c_void_p(flatten_twa.ctypes.data),  # Time-space twa
        ctypes.c_void_p(flatten_tws.ctypes.data),  # Time-space tws
        ctypes.c_void_p(flatten_wa.ctypes.data),  # Time-space twa
        ctypes.c_void_p(flatten_hs.ctypes.data),  # Time-space tws

        ctypes.c_void_p(twa_coords.ctypes.data),
        ctypes.c_void_p(tws_coords.ctypes.data),
        ctypes.c_void_p(wa_coords.ctypes.data),
        ctypes.c_void_p(hs_coords.ctypes.data),
        ctypes.c_void_p(modes_speed.ctypes.data),  # Speed for each mode
        ctypes.c_void_p(modes_power.ctypes.data),  # Power for each mode
        ctypes.c_size_t(modes_shape0),  # Number of modes
        ctypes.c_size_t(modes_shape1),  # Mode index 1
        ctypes.c_size_t(modes_shape2),  # Mode index 24
        ctypes.c_size_t(modes_shape3),
        ctypes.c_size_t(modes_shape4),
        

        ctypes.c_size_t(start_point),  # Starting point
        ctypes.c_size_t(end_point),  # Ending point
        ctypes.c_size_t(n_time_steps),  # Number of time steps
        ctypes.c_double(time_res),  # Time resolution
        ctypes.c_double(max_total_energy),  # Maximum energy
    )
    return finalres, lng, lat, index2mode

def pythonify_output(finalres, lng, lat, index2mode):
    num_solutions = finalres.num_solutions
    costs = np.ctypeslib.as_array(finalres.costs, shape=(num_solutions,))
    durations = np.ctypeslib.as_array(finalres.durations, shape=(num_solutions,))
    solutions_sizes = np.ctypeslib.as_array(finalres.solutions_sizes, shape=(num_solutions,))
    operationpoints = np.sum(solutions_sizes)
    places = np.ctypeslib.as_array(finalres.places, shape=(operationpoints,))
    times = np.ctypeslib.as_array(finalres.times, shape=(operationpoints,))
    modes = np.ctypeslib.as_array(finalres.modes, shape=(operationpoints,))

    places_arr = np.empty(shape=(num_solutions,), dtype=list)
    modes_arr = np.empty(shape=(num_solutions,), dtype=list)
    times_arr = np.empty(shape=(num_solutions,), dtype=list)
    for i in range(num_solutions):
        solution_slice = range(np.sum(solutions_sizes[:i]), np.sum(solutions_sizes[:i + 1]))
        places_arr[i] = [(lng[place], lat[place]) for place in places[solution_slice]]
        modes_arr[i] = [index2mode[mode] for mode in modes[solution_slice][:-1]]
        times_arr[i] = times[solution_slice]

    return costs, durations, places_arr, times_arr, modes_arr

def remove_dominated_solutions(costs, durations, places, times, modes):
    last_index = np.argmin(costs) + 1

    return (costs[:last_index],
            durations[:last_index],
            places[:last_index],
            times[:last_index],
            modes[:last_index]
            )