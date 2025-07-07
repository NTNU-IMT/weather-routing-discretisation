// Copyright (c) 2025, NTNU
// Authors: Aurore Wendling <aurore.wendling@ntnu.no>, Alberto Tamburini <albetam@dtu.dk>
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

pub mod findroutes;
pub mod findroutes_waves;

use std::f64::consts::PI;

pub fn backtrack_routes(
    cost: &Vec<Vec<f64>>,
    pred_place: &Vec<Vec<usize>>,
    pred_time: &Vec<Vec<usize>>,
    mode: &Vec<Vec<usize>>,
    stop_time: &[f64],
    start_point: usize,
    end_point: usize,
    time_resolution: f64,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    usize,
) {
    let route_times: Vec<usize> = cost[end_point]
        .iter()
        .enumerate()
        .filter(|&(_, x)| x.is_finite())
        .map(|(i, _)| i)
        .collect();
    let num_routes: usize = route_times.len();
    let mut cost_res: Vec<f64> = Vec::with_capacity(num_routes);
    let mut duration_res: Vec<f64> = Vec::with_capacity(num_routes);
    let mut solution_sizes: Vec<usize> = Vec::with_capacity(num_routes);
    let mut pred_place_res: Vec<usize> = Vec::with_capacity(num_routes);
    let mut pred_time_res: Vec<usize> = Vec::with_capacity(num_routes);
    let mut mode_res: Vec<usize> = Vec::with_capacity(num_routes);
    for end_time in route_times {
        cost_res.push(cost[end_point][end_time]);
        duration_res.push((end_time as f64) * time_resolution + stop_time[end_point]);

        let mut pred_place_temp: Vec<usize> = Vec::new();
        let mut pred_time_temp: Vec<usize> = Vec::new();
        let mut mode_temp: Vec<usize> = Vec::new();
        pred_place_temp.push(end_point);
        pred_time_temp.push(end_time);
        mode_temp.push(usize::MAX);
        loop {
            let iter_place: usize = *pred_place_temp.last().unwrap();
            let iter_time: usize = *pred_time_temp.last().unwrap();
            if iter_place == start_point {
                break;
            } else {
                pred_place_temp.push(pred_place[iter_place][iter_time]);
                pred_time_temp.push(pred_time[iter_place][iter_time]);
                mode_temp.push(mode[iter_place][iter_time]);
            }
        }
        pred_place_temp.reverse();
        pred_time_temp.reverse();
        mode_temp.reverse();
        solution_sizes.push(pred_place_temp.len());
        pred_place_res.append(&mut pred_place_temp);
        pred_time_res.append(&mut pred_time_temp);
        mode_res.append(&mut mode_temp);
    }

    return (
        cost_res,
        duration_res,
        pred_place_res,
        pred_time_res,
        mode_res,
        solution_sizes,
        num_routes,
    );
}

pub fn init_matrices(
    graph: &Vec<&[usize]>,
    time_steps: usize,
    start_point: usize,
) -> (
    Vec<Vec<f64>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
) {
    // Point-time cost
    let mut cost: Vec<Vec<f64>> = Vec::with_capacity(graph.len());
    for _ in 0..graph.len() {
        cost.push(vec![f64::INFINITY; time_steps]);
    }
    cost[start_point] = vec![0.0; time_steps];

    // Place predecessor indexes
    let mut pred_place: Vec<Vec<usize>> = Vec::with_capacity(graph.len());
    for _ in 0..graph.len() {
        pred_place.push(vec![usize::MAX; time_steps]);
    }
    pred_place[start_point] = vec![0; time_steps];

    // Time predecessor indexes
    let mut pred_time: Vec<Vec<usize>> = Vec::with_capacity(graph.len());
    for _ in 0..graph.len() {
        pred_time.push(vec![usize::MAX; time_steps]);
    }
    pred_time[start_point] = vec![0; time_steps];

    // Point-time Modes indexes
    let mut mode: Vec<Vec<usize>> = Vec::with_capacity(graph.len());
    for _ in 0..graph.len() {
        mode.push(vec![usize::MAX; time_steps]);
    }
    mode[start_point] = vec![0; time_steps];

    return (cost, pred_place, pred_time, mode);
}

#[inline]
pub fn weather(
    angle: &Vec<&[f64]>,
    lng: &[f64],
    lat: &[f64],
    current: usize,
    next: usize,
    t_current: usize,
) -> f64 {
    return input_angle(
        lng[current],
        lat[current],
        lng[next],
        lat[next],
        angle[current][t_current],
    );
}

#[inline]
pub fn input_angle(from_lng: f64, from_lat: f64, to_lng: f64, to_lat: f64, true_angle: f64) -> f64 {
    return (true_angle - bearing(from_lng, from_lat, to_lng, to_lat)) % (2.0 * PI);
}

#[inline]
pub fn bearing(from_lng: f64, from_lat: f64, to_lng: f64, to_lat: f64) -> f64 {
    let start_lon: f64 = if from_lng < 0.0 {
        2.0 * PI + from_lng
    } else {
        from_lng
    };
    let end_lon: f64 = if to_lng < 0.0 {
        2.0 * PI + to_lng
    } else {
        to_lng
    };
    let dlon: f64 = end_lon - start_lon;
    return (dlon.sin() * to_lat.cos())
        .atan2(from_lat.cos() * to_lat.sin() - from_lat.sin() * to_lat.cos() * dlon.cos());
}

#[inline]
pub fn shortest_distance(from_lng: f64, from_lat: f64, to_lng: f64, to_lat: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6378137.0;
    return EARTH_RADIUS_M * shortest_angular_distance(from_lng, from_lat, to_lng, to_lat);
}

#[inline]
pub fn shortest_angular_distance(from_lng: f64, from_lat: f64, to_lng: f64, to_lat: f64) -> f64 {
    return (from_lat.sin() * to_lat.sin()
        + from_lat.cos() * to_lat.cos() * (to_lng - from_lng).cos())
    .acos();
}
