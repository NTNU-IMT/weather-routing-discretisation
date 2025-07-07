# Copyright (c) 2025, NTNU
# Author: Aurore Wendling <aurore.wendling@ntnu.no>
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

import cdsapi
from zipfile import ZipFile
import os.path
import xarray as xr

def download_hindcast_data(
    year: int):
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind"
        ],
        "year": [str(year)],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [90, -180, -90, 180],
        "grid": [0.25, 0.25]
    }

    client = cdsapi.Client()
    target = str(year) + '-1h-025deg.zip'
    client.retrieve(dataset, request, target)

    with ZipFile(target) as zObject:
        zObject.extractall(
            path='./input/weather')

    os.rename('./input/weather/data_stream-oper_stepType-instant.nc',
              './input/weather/' + str(year) + '-world-wind-1h-025deg.nc')
    os.remove(target)

if __name__ == '__main__':
    
    year = 2023

    if not os.path.isfile('./input/weather/' + str(year) + '-world-wind-1h-025deg.nc'):
        download_hindcast_data(year)