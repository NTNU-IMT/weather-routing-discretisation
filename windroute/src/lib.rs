// Copyright (c) 2025, NTNU
// Authors: Aurore Wendling <aurore.wendling@ntnu.no>, Alberto Tamburini <albetam@dtu.dk>
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

use core::slice;

mod paretosearch;
mod routes;

use crate::routes::Routes;

#[unsafe(no_mangle)]
pub extern "C" fn graph_search(
    cgraph: *const usize,
    cstop_time: *const f64,
    cgraph_len: *const usize,
    cgraph_len_len: usize,
    clng: *const f64,
    clat: *const f64,
    cis_reachable: *const bool,
    cweather_twa: *const f64,
    cweather_tws: *const f64,
    ctwa_coords: *const f64,
    ctws_coords: *const f64,
    cmodes_speed: *const f64,
    cmodes_power: *const f64,
    cmodes_shape0: usize,
    cmodes_shape1: usize,
    cmodes_shape2: usize,
    start_point: usize,
    end_point: usize,
    time_steps: usize,
    time_resolution: f64,
    max_energy: f64,
) -> Routes {
    let cgraph_sizes: &[usize] = unsafe { slice::from_raw_parts(cgraph_len, cgraph_len_len) };
    let graph: Vec<&[usize]> = slice_vector_of_vector(cgraph, cgraph_sizes);
    let stop_time: &[f64] = unsafe { slice::from_raw_parts(cstop_time, graph.len()) };

    let lng: &[f64] = unsafe { slice::from_raw_parts(clng, cgraph_len_len) };
    let lat: &[f64] = unsafe { slice::from_raw_parts(clat, cgraph_len_len) };
    let is_reachable: Vec<&[bool]> = slice_matrix(&cis_reachable, graph.len(), time_steps);

    let weather_twa: Vec<&[f64]> = slice_matrix(&cweather_twa, graph.len(), time_steps);
    let weather_tws: Vec<&[f64]> = slice_matrix(&cweather_tws, graph.len(), time_steps);
    let twa_coords: &[f64] = unsafe { slice::from_raw_parts(ctwa_coords, cmodes_shape2) };
    let tws_coords: &[f64] = unsafe { slice::from_raw_parts(ctws_coords, cmodes_shape1) };

    let modes_speed: Vec<Vec<&[f64]>> =
        slice_3dmatrix(&cmodes_speed, cmodes_shape0, cmodes_shape1, cmodes_shape2);
    let modes_power: Vec<Vec<&[f64]>> =
        slice_3dmatrix(&cmodes_power, cmodes_shape0, cmodes_shape1, cmodes_shape2);

    return Routes::new(paretosearch::findroutes::generate(
        &graph,
        stop_time,
        lng,
        lat,
        &is_reachable,
        &weather_twa,
        &weather_tws,
        twa_coords,
        tws_coords,
        &modes_speed,
        &modes_power,
        start_point,
        end_point,
        time_steps,
        time_resolution,
        max_energy,
    ));
}

#[unsafe(no_mangle)]
pub extern "C" fn graph_search_waves(
    cgraph: *const usize,
    cstop_time: *const f64,
    cgraph_len: *const usize,
    cgraph_len_len: usize,
    clng: *const f64,
    clat: *const f64,
    cis_reachable: *const bool,
    cweather_twa: *const f64,
    cweather_tws: *const f64,
    cweather_wd: *const f64,
    cweather_hs: *const f64,
    ctwa_coords: *const f64,
    ctws_coords: *const f64,
    ciwd_coords: *const f64,
    chs_coords: *const f64,
    cmodes_speed: *const f64,
    cmodes_power: *const f64,
    cmodes_shape0: usize,
    cmodes_shape1: usize,
    cmodes_shape2: usize,
    cmodes_shape3: usize,
    cmodes_shape4: usize,
    start_point: usize,
    end_point: usize,
    time_steps: usize,
    time_resolution: f64,
    max_energy: f64,
) -> Routes {
    let cgraph_sizes: &[usize] = unsafe { slice::from_raw_parts(cgraph_len, cgraph_len_len) };
    let graph: Vec<&[usize]> = slice_vector_of_vector(cgraph, cgraph_sizes);
    let stop_time: &[f64] = unsafe { slice::from_raw_parts(cstop_time, graph.len()) };

    let lng: &[f64] = unsafe { slice::from_raw_parts(clng, cgraph_len_len) };
    let lat: &[f64] = unsafe { slice::from_raw_parts(clat, cgraph_len_len) };
    let is_reachable: Vec<&[bool]> = slice_matrix(&cis_reachable, graph.len(), time_steps);

    let weather_twa: Vec<&[f64]> = slice_matrix(&cweather_twa, graph.len(), time_steps);
    let weather_tws: Vec<&[f64]> = slice_matrix(&cweather_tws, graph.len(), time_steps);
    let weather_wd: Vec<&[f64]> = slice_matrix(&cweather_wd, graph.len(), time_steps);
    let weather_hs: Vec<&[f64]> = slice_matrix(&cweather_hs, graph.len(), time_steps);
    let twa_coords: &[f64] = unsafe { slice::from_raw_parts(ctwa_coords, cmodes_shape2) };
    let tws_coords: &[f64] = unsafe { slice::from_raw_parts(ctws_coords, cmodes_shape1) };
    let iwd_coords: &[f64] = unsafe { slice::from_raw_parts(ciwd_coords, cmodes_shape3) };
    let hs_coords: &[f64] = unsafe { slice::from_raw_parts(chs_coords, cmodes_shape4) };

    let modes_speed: Vec<Vec<Vec<Vec<&[f64]>>>> = slice_5dmatrix(
        &cmodes_speed,
        cmodes_shape0,
        cmodes_shape1,
        cmodes_shape2,
        cmodes_shape3,
        cmodes_shape4,
    );
    let modes_power: Vec<Vec<Vec<Vec<&[f64]>>>> = slice_5dmatrix(
        &cmodes_power,
        cmodes_shape0,
        cmodes_shape1,
        cmodes_shape2,
        cmodes_shape3,
        cmodes_shape4,
    );
    return Routes::new(paretosearch::findroutes_waves::generate(
        &graph,
        stop_time,
        lng,
        lat,
        &is_reachable,
        &weather_twa,
        &weather_tws,
        &weather_wd,
        &weather_hs,
        twa_coords,
        tws_coords,
        iwd_coords,
        hs_coords,
        &modes_speed,
        &modes_power,
        start_point,
        end_point,
        time_steps,
        time_resolution,
        max_energy,
    ));
}

#[unsafe(no_mangle)]
pub extern "C" fn free_routes(routes: Routes) {
    unsafe { routes.free() };
}

fn slice_vector_of_vector<T>(to_slice: *const T, vector_sizes: &[usize]) -> Vec<&[T]> {
    let slice_temp: &[T] = unsafe { slice::from_raw_parts(to_slice, vector_sizes.iter().sum()) };
    let mut res: Vec<&[T]> = Vec::with_capacity(vector_sizes.len());
    let mut added: usize = 0;
    for i in 0..vector_sizes.len() {
        res.push(&slice_temp[added..(added + vector_sizes[i])]);
        added += vector_sizes[i];
    }

    return res;
}

fn slice_matrix<T>(to_slice: &*const T, shape0: usize, shape1: usize) -> Vec<&[T]> {
    let slice_temp: &[T] = unsafe { slice::from_raw_parts(*to_slice, shape0 * shape1) };
    let mut res: Vec<&[T]> = Vec::with_capacity(shape0);
    for i in 0..shape0 {
        res.push(&slice_temp[(i * shape1)..((i + 1) * shape1)]);
    }

    return res;
}

fn slice_3dmatrix<T>(
    to_slice: &*const T,
    shape0: usize,
    shape1: usize,
    shape2: usize,
) -> Vec<Vec<&[T]>> {
    let slice_temp: &[T] = unsafe { slice::from_raw_parts(*to_slice, shape0 * shape1 * shape2) };
    let mut res: Vec<Vec<&[T]>> = Vec::with_capacity(shape0);
    for i in 0..shape0 {
        let mut res_i: Vec<&[T]> = Vec::with_capacity(shape1);
        for j in 0..shape1 {
            res_i.push(
                &slice_temp
                    [(i * shape1 * shape2 + j * shape2)..(i * shape1 * shape2 + (j + 1) * shape2)],
            );
        }
        res.push(res_i);
    }

    return res;
}

fn slice_5dmatrix<T>(
    to_slice: &*const T,
    shape0: usize,
    shape1: usize,
    shape2: usize,
    shape3: usize,
    shape4: usize,
) -> Vec<Vec<Vec<Vec<&[T]>>>> {
    let slice_temp: &[T] = unsafe { slice::from_raw_parts(*to_slice, shape0 * shape1 * shape2 * shape3 * shape4) };
    let mut res: Vec<Vec<Vec<Vec<&[T]>>>> = Vec::with_capacity(shape0);
    for i in 0..shape0 {
        let mut res_i: Vec<Vec<Vec<&[T]>>> = Vec::with_capacity(shape1);
        for j in 0..shape1 {
            let mut res_j: Vec<Vec<&[T]>> = Vec::with_capacity(shape2);
            for k in 0..shape2 {
                let mut res_k: Vec<&[T]> = Vec::with_capacity(shape3);
                for l in 0..shape3 {
                    res_k.push(
                        &slice_temp[(i * shape1 * shape2 * shape3 * shape4
                            + j * shape2 * shape3 * shape4
                            + k * shape3 * shape4
                            + l * shape4)
                            ..(i * shape1 * shape2 * shape3 * shape4
                                + j * shape2 * shape3 * shape4
                                + k * shape3 * shape4
                                + (l + 1) * shape4)],
                    );
                }
                res_j.push(res_k);
            }
            res_i.push(res_j);
        }
        res.push(res_i);
    }

    return res;
}
