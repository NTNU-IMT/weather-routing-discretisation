# Copyright (c) 2025, NTNU
# Author: Aurore Wendling <aurore.wendling@ntnu.no>
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

import math
import warnings

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import matplotlib.animation as animation
import ffmpeg

from .tools import *
import xarray as xr

from scipy.optimize import curve_fit

def plot_map(title:str,ax=None, llcrnrlon=-180., llcrnrlat=-80., urcrnrlon=180., urcrnrlat=80.,
                rsphere=(6378137.00, 6356752.3142),
                resolution='c', projection='merc',
                lat_ts=20.,):
    fig = None
    if not ax:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # setup mercator map projection.
    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                rsphere=rsphere,
                resolution=resolution, projection=projection,
                lat_ts=lat_ts, ax=ax)
    m.drawcoastlines()
    m.fillcontinents()
    m.drawparallels(np.arange(-90, 90, 20), labels=[1, 1, 0, 1])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])
    ax.set_title(title)
    return m,ax,fig

def plot_wind(m: Basemap,
                  weather_file: str,
                time: datetime):
    ds = xr.open_dataset(weather_file)
    if 'time' in ds.dims:
        ds = ds.rename({'time': 'valid_time'})

    x = ds.sel(valid_time=time).longitude.values
    y = ds.sel(valid_time=time).latitude.values
    u = ds.sel(valid_time=time).u10.values
    v = ds.sel(valid_time=time).v10.values
    speed = np.sqrt(u * u + v * v)

    xx, yy = np.meshgrid(x, y)
    m.pcolormesh(xx, yy, speed, cmap=plt.cm.viridis, latlon=True)
    res = 20
    m.quiver(xx[::res, ::res], yy[::res, ::res], u[::res, ::res], v[::res, ::res], scale=800, width=0.0007,
             units='width', latlon=True)
    m.drawcoastlines()
    m.fillcontinents()

def animate_wind_map(fig,
                     m: Basemap,
                     weather_file: str,
                     start_time: datetime,
                     route: list,
                     route_stats:list):
    ds = xr.open_dataset(weather_file)


    x = ds.sel(valid_time=start_time).longitude.values
    y = ds.sel(valid_time=start_time).latitude.values
    xx, yy = np.meshgrid(x, y)
    quiver_res = 20

    u = []
    v = []
    speed = []
    for stage in range(len(route)-1):
        current_time = start_time + timedelta(hours=route_stats[stage]['time [h]'])
        u.append(ds.sel(valid_time=current_time, method='nearest').u10.values)
        v.append(ds.sel(valid_time=current_time, method='nearest').v10.values)
        speed.append(np.sqrt(u[stage] * u[stage] + v[stage] * v[stage]))
        print(stage)

    speed_plot = m.pcolormesh(xx, yy, speed[0], cmap=plt.cm.viridis, latlon=True)
    vector_plot = m.quiver(xx[::quiver_res, ::quiver_res], yy[::quiver_res, ::quiver_res], u[0][::quiver_res, ::quiver_res], v[0][::quiver_res, ::quiver_res], scale=800, width=0.0007,
            units='width', latlon=True)
    m.drawcoastlines()
    m.fillcontinents()
    plot_route(m, route, color='r', linewidth=2)
    point = m.plot(x=math.degrees(route[0][0]), y=math.degrees(route[0][1]), marker='o', markersize=5, color='r', latlon=True)[0]

    def update_map(stage):
        speed_plot.set_array(speed[stage])
        vector_plot.set_UVC(u[stage][::quiver_res, ::quiver_res], v[stage][::quiver_res, ::quiver_res])
        point.set_data(m(math.degrees(route[stage][0]), math.degrees(route[stage][1])))
        return point,speed_plot, vector_plot

    anim = animation.FuncAnimation(fig=fig, func=update_map, frames=len(route)-1, interval=1000)
    return anim

def plot_route(m:'Basemap',
               places,
               label='',
               color='k',
               linewidth=0.5):
    x = [math.degrees(waypoint[0]) for waypoint in places]
    y = [math.degrees(waypoint[1]) for waypoint in places]

    lims = [-180, 180]
    for i in range(len(x)):
        if x[i] < -180:
            x[i] = x[i]%180
        if x[i] > 180:
            x[i] = x[i]%180 -180
    split_idx = 0
    for i in range(len(x)-1):
        if abs(x[i]-x[i+1]) >= 90:
            split_idx = i+1
            break
    m.plot(x=x[:split_idx],
               y=y[:split_idx],
               marker=None, color=color,
               linewidth=linewidth, latlon=True,label='_nolegend_')
    m.plot(x=x[split_idx:],
               y=y[split_idx:],
               marker=None, color=color,
               linewidth=linewidth, latlon=True,label='_nolegend_')
    for slc in unlink_wrap(x, lims):
        m.plot(x=x[slc],
               y=y[slc],
               marker=None, color=color,
               linewidth=linewidth, latlon=True,label=label)
        label = '_nolegend_'
    plt.legend()
    
def plot_grid(m:'Basemap', grid: np.ndarray):
    m.scatter(x=[math.degrees(grid[i,j][0]) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j]],
              y=[math.degrees(grid[i,j][1]) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j]],
              marker='o', s=1, color='k', latlon=True)

def plot_route_distribution(costs, durations, ax, label=None, title='Pareto route distribution', plot_func=plt.scatter, plot_args=(2, None, 'o'), color=None):
    plot_func(durations, costs, label=label, *plot_args, color=color)
    ax.set_xlabel('Voyage time [h]')
    ax.set_ylabel('Energy [kWh]')
    ax.set_title(title)

def unlink_wrap(dat, lims=[-np.pi, np.pi], thresh = 0.95):
    """
    Iterate over contiguous regions of `dat` (i.e. where it does not
    jump from near one limit to the other).

    This function returns an iterator object that yields slice
    objects, which index the contiguous portions of `dat`.

    This function implicitly assumes that all points in `dat` fall
    within `lims`.

    """
    jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))





