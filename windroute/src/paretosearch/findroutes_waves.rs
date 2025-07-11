// Copyright (c) 2025, NTNU
// Authors: Aurore Wendling <aurore.wendling@ntnu.no>, Alberto Tamburini <albetam@dtu.dk>
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

use crate::paretosearch;

pub fn generate(
    graph: &Vec<&[usize]>,
    stop_time: &[f64],
    lng: &[f64],
    lat: &[f64],
    is_reachable: &Vec<&[bool]>,
    weather_twa: &Vec<&[f64]>,
    weather_tws: &Vec<&[f64]>,
    weather_wd: &Vec<&[f64]>,
    weather_hs: &Vec<&[f64]>,
    twa_coords: &[f64],
    tws_coords: &[f64],
    iwd_coords: &[f64],
    hs_coords: &[f64],
    modes_speed: &Vec<Vec<Vec<Vec<&[f64]>>>>,
    modes_power: &Vec<Vec<Vec<Vec<&[f64]>>>>,
    start_point: usize,
    end_point: usize,
    time_steps: usize,
    time_resolution: f64,
    max_energy: f64,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    usize,
) {
    let (mut cost, mut pred_place, mut pred_time, mut mode) =
        paretosearch::init_matrices(graph, time_steps, start_point);
    for current in start_point..end_point {
        for next_index in 0..graph[current].len() {
            for t_current in 0..time_steps {
                if cost[current][t_current].is_finite() {
                    assert!(
                        weather_twa[current][t_current].is_finite(),
                        "Invalid weather parameter: twa[{}, {}] = {}",
                        current,
                        t_current,
                        weather_twa[current][t_current]
                    );
                    assert!(
                        weather_tws[current][t_current].is_finite(),
                        "Invalid weather parameter: tws[{}, {}] = {}",
                        current,
                        t_current,
                        weather_tws[current][t_current]
                    );
                    assert!(
                        weather_wd[current][t_current].is_finite(),
                        "Invalid weather parameter: wd[{}, {}] = {}",
                        current,
                        t_current,
                        weather_wd[current][t_current]
                    );
                    assert!(
                        weather_hs[current][t_current].is_finite(),
                        "Invalid weather parameter: hs[{}, {}] = {}",
                        current,
                        t_current,
                        weather_hs[current][t_current]
                    );
                    let next: usize = graph[current][next_index];
                    let twa: f64 =
                        paretosearch::weather(weather_twa, lng, lat, current, next, t_current);
                    let tws: f64 = weather_tws[current][t_current];
                    let iwd: f64 =
                        paretosearch::weather(weather_wd, lng, lat, current, next, t_current);
                    let hs: f64 = weather_hs[current][t_current];
                    for mode_index in 0..modes_speed.len() {
                        let (energy_kwh, duration, _speed) = sail_arc(
                            &modes_speed[mode_index],
                            &modes_power[mode_index],
                            lng[current],
                            lat[current],
                            lng[next],
                            lat[next],
                            twa_coords
                                .iter()
                                .enumerate()
                                .map(|(i, &a)| (i, (a - twa).abs()))
                                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(i, _)| i)
                                .unwrap(),
                            tws_coords
                                .iter()
                                .enumerate()
                                .map(|(i, &a)| (i, (a - tws).abs()))
                                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(i, _)| i)
                                .unwrap(),
                            iwd_coords
                                .iter()
                                .enumerate()
                                .map(|(i, &a)| (i, (a - iwd).abs()))
                                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(i, _)| i)
                                .unwrap(),
                            hs_coords
                                .iter()
                                .enumerate()
                                .map(|(i, &a)| (i, (a - hs).abs()))
                                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(i, _)| i)
                                .unwrap(),
                            stop_time[current],
                        );
                        if duration.is_finite() {
                            let t_next: usize = (((time_resolution * (t_current as f64) + duration)
                                / time_resolution)
                                .round() as usize)
                                .min(time_steps - 1);
                            if is_reachable[next][t_next] {
                                if cost[current][t_current] + energy_kwh < cost[next][t_next]
                                    && cost[current][t_current] + energy_kwh < max_energy
                                {
                                    cost[next][t_next] = cost[current][t_current] + energy_kwh;
                                    pred_place[next][t_next] = current;
                                    pred_time[next][t_next] = t_current;
                                    mode[next][t_next] = mode_index;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return paretosearch::backtrack_routes(
        &cost,
        &pred_place,
        &pred_time,
        &mode,
        stop_time,
        start_point,
        end_point,
        time_resolution,
    );
}

fn sail_arc(
    mode_speed: &Vec<Vec<Vec<&[f64]>>>,
    mode_power: &Vec<Vec<Vec<&[f64]>>>,
    from_lng: f64,
    from_lat: f64,
    to_lng: f64,
    to_lat: f64,
    weather_j: usize,
    weather_i: usize,
    weather_k: usize,
    weather_l: usize,
    stop_duration: f64,
) -> (f64, f64, f64) {
    if from_lng != to_lng || from_lat != to_lat {
        let speed: f64 = mode_speed[weather_i][weather_j][weather_k][weather_l];
        if speed != 0.0 {
            let length: f64 = paretosearch::shortest_distance(from_lng, from_lat, to_lng, to_lat);
            return (
                mode_power[weather_i][weather_j][weather_k][weather_l] * length / speed / 3600.0,
                stop_duration + length / speed / 3600.0,
                speed,
            );
        } else {
            return (0.0, f64::INFINITY, 0.0);
        }
    } else {
        return (0.0, f64::INFINITY, 0.0);
    }
}
