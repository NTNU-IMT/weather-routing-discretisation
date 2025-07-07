# Copyright (c) 2025, NTNU
# Author: Aurore Wendling <aurore.wendling@ntnu.no>
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

import xarray as xr
import numpy as np

time_res = [12, 8, 6, 4, 3, 2, 1]
space_res = [2, 1, 0.5, 0.25]

for time_res_h in time_res:
    for space_res_deg in space_res:

        dataset = xr.open_dataset('./input/weather/2023-world-wind-1h-025deg.nc')

        time_index = dataset.coords.variables.mapping['valid_time'].values
        dataset = dataset.reindex({"valid_time": time_index[::time_res_h]})

        lon_index = dataset.coords.variables.mapping['longitude'].values
        dataset = dataset.reindex({"longitude": lon_index[::int(space_res_deg/0.25)]})

        lat_index = dataset.coords.variables.mapping['latitude'].values
        dataset = dataset.reindex({"latitude": lat_index[::int(space_res_deg/0.25)]})

        dataset.to_netcdf('./input/weather/reindexed/2023-world-wind-' + str(time_res_h) + 'h-' + str(int(100*space_res_deg)) + 'deg.nc')

        dataset.close()