// Copyright (c) 2025, NTNU
// Authors: Aurore Wendling <aurore.wendling@ntnu.no>, Alberto Tamburini <albetam@dtu.dk>
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

use std::mem;

#[repr(C)]
pub struct Routes {
    costs: *mut f64,
    durations: *mut f64,
    places: *mut usize,
    times: *mut usize,
    modes: *mut usize,
    solutions_sizes: *mut usize,
    num_solutions: usize,
}

impl Routes {
    pub fn new(
        (
            mut cost_res,
            mut duration_res,
            mut pred_place_res,
            mut pred_time_res,
            mut mode_res,
            mut solution_sizes_res,
            num_routes,
        ): (
            Vec<f64>,
            Vec<f64>,
            Vec<usize>,
            Vec<usize>,
            Vec<usize>,
            Vec<usize>,
            usize,
        ),
    ) -> Routes {
        let cost: *mut f64 = cost_res.as_mut_ptr();
        let duration: *mut f64 = duration_res.as_mut_ptr();
        let pred_place: *mut usize = pred_place_res.as_mut_ptr();
        let pred_time: *mut usize = pred_time_res.as_mut_ptr();
        let mode: *mut usize = mode_res.as_mut_ptr();
        let solution_sizes: *mut usize = solution_sizes_res.as_mut_ptr();
        mem::forget(cost_res);
        mem::forget(duration_res);
        mem::forget(pred_place_res);
        mem::forget(pred_time_res);
        mem::forget(mode_res);
        mem::forget(solution_sizes_res);
        return Routes {
            costs: cost,
            durations: duration,
            places: pred_place,
            times: pred_time,
            modes: mode,
            solutions_sizes: solution_sizes,
            num_solutions: num_routes,
        };
    }

    pub unsafe fn free(self) {
        unsafe {
            let costs: &mut [f64] = std::slice::from_raw_parts_mut(self.costs, self.num_solutions);
            let durations: &mut [f64] =
                std::slice::from_raw_parts_mut(self.durations, self.num_solutions);
            let solutions_sizes: &mut [usize] =
                std::slice::from_raw_parts_mut(self.solutions_sizes, self.num_solutions);
            let places: &mut [usize] =
                std::slice::from_raw_parts_mut(self.places, solutions_sizes.iter().sum());
            let times: &mut [usize] =
                std::slice::from_raw_parts_mut(self.times, solutions_sizes.iter().sum());
            let modes: &mut [usize] =
                std::slice::from_raw_parts_mut(self.modes, solutions_sizes.iter().sum());

            mem::drop(Box::from_raw(costs.as_mut_ptr()));
            mem::drop(Box::from_raw(durations.as_mut_ptr()));
            mem::drop(Box::from_raw(places.as_mut_ptr()));
            mem::drop(Box::from_raw(times.as_mut_ptr()));
            mem::drop(Box::from_raw(modes.as_mut_ptr()));
            mem::drop(Box::from_raw(solutions_sizes.as_mut_ptr()));
        }
    }
}
