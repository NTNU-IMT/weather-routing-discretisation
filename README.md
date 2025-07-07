# How to discretise a ship's voyage when optimising for high fractions of wind propulsion?

## Introduction

This repository contains the routing program and test cases used for an article with the same title as this readme document. The purpose is to show the algorithm and specific settings used in the article. 

The article can be found at the following link:

The routing program is written in Python and Rust. It is designed to optimise route and engine power allocation for a ship with primary wind propulsion and uses a grid-based 3D Dynamic Programming method. The test cases are set up to evaluate the effect of discretisation on the solutions.

The program is written in Python and Rust, and released under the [GPL Version 3](https://www.gnu.org/licenses/). It can be used, redistributed and modified under the terms of this license, however note that the code is published for only transparency: it will not be maintained and no support will be provided.

## Requirements

The routing program and test cases can be run on a Linux machine with Python 3.11 and the Rust files are compiled with Rust 1.86.0. The following python packages are required:

- basemap
- matplotlib
- netCDF4
- numba
- numpy
- pandas
- scipy
- searoute
- shapely
- xarray
- cdsapi>=0.7.4

## Folder structure

`routing_func` contains the Python code to discretise the problem into a graph and post-process the results.

`windroute` contains the graph search function written in Rust.

`input/routes` defines the test routes (origin and destination) via jason files

`input/setup` defines default setup parameters via a jason file.

`input/ship` contains the test vessel's speed polars as a netcdf file.

`input/ship` holds weather datasets in netcdf format.

`output` is the destination folder for test results

## Use and reproduction of the experiments

In order to reproduce the tests described in the article, it is sufficient to:

1. Download weather data from the ECMWF climate data store, using the script `download_weather_data.py`. It is necessary to have an account and to follow [the instructions of  the ECMWF climate data store](https://cds.climate.copernicus.eu/how-to-api) to download meteorological data.
2. To test weather data resolution, the weather dataset needs to be reindexed using `reindex_weather_dataset.py`.
3. Run all the convergence tests and write the output to the `output` folder using the script `convergence_study.py`.

## Authors

The software is developed by [Aurore Wendling](https://www.ntnu.edu/employees/aurore.wendling) at the Norwegian University of Science and Technology (NTNU).

## License

The software is released under the [GPL Version 3](https://www.gnu.org/licenses/).

## Acknowledgments

The graph search function was translated to Rust by Alberto Tamburini from the Technical University of Denmark (DTU).

The route optimisation algorithm uses the Python package [searoute-py](https://github.com/genthalili/searoute-py) to find the shortest path between two ports.

This software is developed for a PhD thesis supervised by Benjamin Lagemann at the Norwegian University of Science and Technology (NTNU).