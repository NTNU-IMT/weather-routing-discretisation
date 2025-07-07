# Copyright (c) 2025, NTNU
# Author: Aurore Wendling <aurore.wendling@ntnu.no>
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

import cProfile,pstats

import matplotlib.pyplot as plt
import numpy as np
import os.path
import multiprocessing
from routing_func.RouteOptimisation import *
from routing_func.plotting import *
import pandas as pd
import pickle
from adjustText import adjust_text

def read_setup_file(file):
    with open(file, 'r') as open_file:
        setup_dict = json.load(open_file)
    return setup_dict

def set_parameters(
    departure_time: datetime,
    waypoints: list,
    time_res_h=0.125,
    n_stages=40,
    allowed_deviation_percent=30,
    n_adj_points=2,
    n_side_points=15,
    weather_time_res_h=1,
    weather_space_res_deg=0.25,
    aspect_ratio=2,
    n_time_steps=5000):

    route_length_km = compute_shortest_distance(
        waypoints[0],
        waypoints[-1]
    ) / 1000
    allowed_deviation_km = allowed_deviation_percent * route_length_km / 100
    forward_resolution = route_length_km / n_stages
    if n_side_points=='from forward res':
        side_resolution = forward_resolution / aspect_ratio
        number_of_side_points = int(allowed_deviation_km / side_resolution)
    else:
        side_resolution = allowed_deviation_km / n_side_points
        number_of_side_points = int(allowed_deviation_km / side_resolution)

    mean_speed_kts = 5
    max_voyage_time_h = math.ceil(route_length_km / (mean_speed_kts * 1.852))
    if n_time_steps:
        time_res_h = 1/(2**(round(math.log(1/(max_voyage_time_h / n_time_steps), 2))))

    with open('./input/setup/optimisation_setup_cv.json', 'r') as open_file:
        setup_dict = json.load(open_file)
    setup_dict['waypoints'] = waypoints
    setup_dict['forward resolution km'] = forward_resolution
    setup_dict['side resolution km'] = side_resolution
    setup_dict['number of side points'] = number_of_side_points
    setup_dict['departure time'] = datetime.strftime(departure_time,'%Y-%m-%d %H:%M')
    setup_dict['latest arrival time'] = datetime.strftime(departure_time + timedelta(hours=max_voyage_time_h),'%Y-%m-%d %H:%M')
    setup_dict['number of adjacent points'] = n_adj_points
    setup_dict['time discretization hours'] = time_res_h
    setup_dict['wind file'] = './input/weather/reindexed/2023-world-wind-' + str(weather_time_res_h) + 'h-' + str(int(100*weather_space_res_deg)) + 'deg.nc'
    return  setup_dict

def run_case(
    setup_dict,
    case_name: str):

    if not os.path.isfile('./output/convergence_study/' + case_name):
        optimisation_instance = RouteOptimisation.from_dict(setup_dict)
        optimisation_instance.discretise(routing_mode='great circle', skip_courant_above_05=True, autocorrect_courant=False)
        optimisation_instance.read_weather(include_waves=False)
        costs, durations, places, times, modes = optimisation_instance.find_routes()

        solutions = pd.DataFrame(
            {
                'costs': costs,
                'durations': durations,
                'places': list(places),
                'times': list(times),
                'modes': list(modes),
            }
        )
        with open('./output/convergence_study/' + case_name, 'wb') as file:
            pickle.dump(solutions, file)

def plot_pareto(
    solutions: pd.DataFrame,
    label: str,
    color: str):
    plt.plot(solutions.durations, solutions.costs, label=label, color=color, linewidth=1)

def fit_results(param_values, solutions,duration):
    def f(param_value, exact_solution, alpha_coeff, p_exponent):
        return exact_solution + alpha_coeff * (param_value ** p_exponent)
    def f1(param_value, exact_solution, alpha_coeff):
        return exact_solution + alpha_coeff * param_value
    def f2(param_value, exact_solution, alpha_coeff):
        return exact_solution + alpha_coeff * (param_value ** 2)
    def f12(param_value, exact_solution, alpha_coeff_1, alpha_coeff_2):
        return exact_solution + alpha_coeff_1 * param_value + alpha_coeff_2 * (param_value ** 2)

    fit_failed = False
    try:
        popt, pcov = curve_fit(f, param_values, solutions, maxfev=50000)
    except RuntimeError:
        fit_failed = True
    if fit_failed or popt[2] < 0.5:
        popt1, pcov1 = curve_fit(f1, param_values, solutions, maxfev=50000)
        l = [(f1(param_values[i], *popt1)-solutions[i])**2 for i in range(len(solutions))]
        standard_dev1 = math.sqrt(sum(l) / len(l))

        popt2, pcov2 = curve_fit(f2, param_values, solutions, maxfev=50000)
        l = [(f2(param_values[i], *popt2)-solutions[i])**2 for i in range(len(solutions))]
        standard_dev2 = math.sqrt(sum(l) / len(l))

        popt12, pcov12 = curve_fit(f12, param_values, solutions, maxfev=50000)
        l = [(f12(param_values[i], *popt12)-solutions[i])**2 for i in range(len(solutions))]
        standard_dev12 = math.sqrt(sum(l) / len(l))

        if standard_dev1 == min(standard_dev1, standard_dev2, standard_dev12):
            selected_fit = f1
            fit_params = popt1
        if standard_dev2 == min(standard_dev1, standard_dev2, standard_dev12):
            selected_fit = f2
            fit_params = popt2
        if standard_dev12 == min(standard_dev1, standard_dev2, standard_dev12):
            selected_fit = f12
            fit_params = popt12
    elif popt[2] > 2:
        popt1, pcov1 = curve_fit(f1, param_values, solutions, maxfev=50000)
        l = [(f1(param_values[i], *popt1)-solutions[i])**2 for i in range(len(solutions))]
        standard_dev1 = math.sqrt(sum(l) / len(l))

        popt2, pcov2 = curve_fit(f2, param_values, solutions, maxfev=50000)
        l = [(f2(param_values[i], *popt2)-solutions[i])**2 for i in range(len(solutions))]
        standard_dev2 = math.sqrt(sum(l) / len(l))

        if standard_dev1 == min(standard_dev1, standard_dev2):
            selected_fit = f1
            fit_params = popt1
        if standard_dev2 == min(standard_dev1, standard_dev2):
            selected_fit = f2
            fit_params = popt2
    else:
        selected_fit = f
        fit_params = popt
    return fit_params, selected_fit

def plot_eca_curve(list, title):
    d = {}
    for solutions, param_value, label, color in list:
        for i in range(len(solutions.durations)):
            d[solutions.durations[i]] = d[solutions.durations[i]] + [(param_value, solutions.costs[i])] if solutions.durations[i] in d else [(param_value, solutions.costs[i])]
    xdata = []
    ydata = {param_value:[] for solutions, param_value, label, color in list}
    ydata2 = []
    fastest_route_duration = math.inf
    fastest_route_cost_th = 0
    for duration, result in d.items():
        x = []
        y = []
        for res in result:
            x += [res[0]]
            y += [res[1]]
        if len(x)==len(list):
            popt, func = fit_results(x,y, duration)
            xdata += [duration]
            if duration < fastest_route_duration:
                fastest_route_duration = duration
                fastest_route_cost_th = popt[0]
            for solutions, param_value, label, color in list:
                ydata[param_value] += [100 * (func(param_value, *popt)-popt[0]) if popt[0] else None]
    xdata = sorted(xdata)
    ydata2 = [val for _, val in sorted(zip(xdata, ydata2))]
    fig, ax = plt.subplots()
    for solutions, param_value, label, color in list:
        plt.plot(xdata,[val/fastest_route_cost_th for _, val in sorted(zip(xdata, ydata[param_value]))], label=label, color=color)
    ax.set_xlabel('Voyage time [h]')
    ax.set_ylabel('Error estimate [% of fastest route cost]')
    ax.legend(title=title)

def plot_error_wrt_best_res(list, title):
    d = {}
    for solutions, param_value, label, color in list:
        for i in range(len(solutions.durations)):
            d[solutions.durations[i]] = d[solutions.durations[i]] + [(param_value, solutions.costs[i])] if solutions.durations[i] in d else [(param_value, solutions.costs[i])]
    xdata = []
    ydata = {param_value:[] for solutions, param_value, label, color in list}
    min_duration = min([k for k in d.keys() if len(d[k])>=len(list)])
    max_cost = d[min_duration][-1][1]
    for duration, result in d.items():
        x = []
        y = []
        for res in result:
            x += [res[0]]
            y += [res[1]]
        if len(x)>=len(list):
            xdata += [duration]
            for i in range(len(list)):
                ydata[x[i]] += [100 * (y[i]-y[-1]) / max_cost]
    xdata = sorted(xdata)
    fig, ax = plt.subplots()
    for solutions, param_value, label, color in list:
        plt.plot(xdata,[val for _, val in sorted(zip(xdata, ydata[param_value]))], label=label, color=color)
    ax.set_xlabel('Voyage time [h]')
    ax.set_ylabel('Error WRT best resolution [% of fastest route cost]')
    ax.legend(title=title)

def make_solutions_list(cases, param_list):
    with multiprocessing.Pool() as pool:
                pool.starmap(run_case, cases)
                pool.close()
                pool.join()

    solutions = []
    new_param_list = []
    for i in range(len(param_list)):
        with open('./output/convergence_study/' + cases[i][1], 'rb') as file:
            s = pickle.load(file)
        if s.costs.any():
            solutions.append(s)
            new_param_list.append(param_list[i])
    
    return solutions, new_param_list

if __name__ == '__main__':
    multiprocessing.freeze_support()

    route_dict = {}

    with open("./input/routes/brest-newyork.json", 'r') as openfile:
        wp = json.load(openfile)
    route_dict['transatlantic'] = [(math.radians(v['lon']), math.radians(v['lat'])) for v in wp]
    with open("./input/routes/transpacific.json", 'r') as openfile:
        wp = json.load(openfile)
    route_dict['transpacific'] = [(math.radians(v['lon']), math.radians(v['lat'])) for v in wp]
    with open("./input/routes/northsea-northbound.json", 'r') as openfile:
        wp = json.load(openfile)
    route_dict['northsea'] = [(math.radians(v['lon']), math.radians(v['lat'])) for v in wp]

    departure_dict = {
        'winter': datetime(year=2023, month=1, day=1, hour=0),
        'summer': datetime(year=2023, month=6, day=1, hour=0)
    }

    lines_formats = {
        'transatlantic - winter': 'C0-o',
        'transatlantic - summer': 'C0-s',
        'transpacific - winter': 'C1-o',
        'transpacific - summer': 'C1-s',
        'northsea - winter': 'C2-o',
        'northsea - summer': 'C2-s'
        }

    route = 'transatlantic'
    season = 'summer'

    # Select the tests to run:
    run_number_of_stages = True
    run_grid_width = True
    run_aspect_ratio = True
    run_aspect_ratio_and_adjacent_points = True
    run_adjacent_points = True
    run_time_resolution = True
    run_weather_time_resolution = True
    run_weather_space_resolution = True

# ---------------------- Number of stages/cells ---------------

    if run_number_of_stages:

        fig1, ax1 = plt.subplots()
        texts = []

        for route in route_dict:
            for season in ['summer', 'winter']:

                n_stages_list = [5, 10, 20, 40, 80, 160]

                cases = []
                for n_stages in n_stages_list:
                    cases += [(
                        set_parameters(
                            departure_time=departure_dict[season],
                            waypoints=route_dict[route],
                            n_stages=n_stages,
                            n_side_points='from forward res'
                            ),
                        'number_of_stages/' + route + '/' + season + '/' + str(n_stages)
                    )]

                solutions, n_stages_list = make_solutions_list(cases, n_stages_list)

                max_cost = max([np.max(s.costs) for s in solutions])
                max_duration = max([np.max(s.durations) for s in solutions])
                min_cost = min([np.min(s.costs) for s in solutions])
                min_duration = min([np.min(s.durations) for s in solutions])

                solutions_list = []
                colors = plt.cm.plasma(np.linspace(0, 1, len(n_stages_list)))

                route_length_km = compute_shortest_distance(route_dict[route][0], route_dict[route][-1]) / 1000
                fig, ax = plt.subplots()
                ax.set_xlabel('Voyage time [h]')
                ax.set_ylabel('Energy [kWh]')
                for i in range(len(solutions)):
                    plot_pareto(solutions[i], str(n_stages_list[i]), colors[i])
                    solutions_list += [(
                        solutions[i], 
                        route_length_km / n_stages_list[i], 
                        str(n_stages_list[i]),
                        colors[i]
                        )]
                ax.legend(title='Number of stages')

                plot_eca_curve(solutions_list, 'Number of stages')

                m, ax = plot_map(None, None)[:2]
                for i in range(len(solutions)):
                    plot_route(m, list(solutions[i].places.iloc[-1]), color=colors[i], linewidth=1, label=str(n_stages_list[i]))
                ax.legend(title='Number of stages')

                hyperareas = []
                for i in range(len(n_stages_list)):
                    hyperareas += [hyperarea(solutions[i].costs, solutions[i].durations, min_cost, min_duration, max_cost, max_duration)]
                ax1.plot(n_stages_list, hyperareas, lines_formats[route + ' - ' + season], label=route + ' - ' + season, markersize=5)

        ax1.set_xlabel('Number of stages []')
        ax1.set_ylabel('Hyperarea []')
        ax1.legend()


# ---------------------- Grid width ---------------------------

    if run_grid_width:

        fig1, ax1 = plt.subplots()

        for route in route_dict:
            for season in ['winter']:

                grid_width_list = [5, 10, 20, 30, 40, 50]

                cases = []
                for grid_width in grid_width_list:
                    cases += [(
                        set_parameters(
                            departure_time=departure_dict[season],
                            waypoints=route_dict[route],
                            allowed_deviation_percent=grid_width,
                            n_side_points='from forward res'
                            ),
                        'max_deviation/' + route + '/' + season + '/' + str(grid_width)
                    )]

                solutions, grid_width_list = make_solutions_list(cases, grid_width_list)
                max_cost = max([np.max(s.costs) for s in solutions])
                max_duration = max([np.max(s.durations) for s in solutions])
                min_cost = min([np.min(s.costs) for s in solutions])
                min_duration = min([np.min(s.durations) for s in solutions])

                solutions_list = []
                colors = plt.cm.plasma(np.linspace(0, 1, len(grid_width_list)))

                fig, ax = plt.subplots()
                ax.set_xlabel('Voyage time [h]')
                ax.set_ylabel('Energy [kWh]')
                for i in range(len(solutions)):
                    plot_pareto(solutions[i], str(grid_width_list[i]), colors[i])
                    solutions_list += [(
                        solutions[i], 
                        grid_width_list[i], 
                        str(grid_width_list[i]),
                        colors[i]
                        )]
                ax.legend(title='Max distance from GCR [% of GCR length]')

                plot_error_wrt_best_res(solutions_list, 'Grid width [% of GCR length]')

                m, ax = plot_map(None, None)[:2]
                for i in range(len(solutions)):
                    plot_route(m, list(solutions[i].places.iloc[-1]), color=colors[i], linewidth=1, label=str(grid_width_list[i]))
                ax.legend(title='Max distance from GCR [% of GCR length]')

                hyperareas = []
                for i in range(len(grid_width_list)):
                    hyperareas += [hyperarea(solutions[i].costs, solutions[i].durations, min_cost, min_duration, max_cost, max_duration)]
                ax1.plot(grid_width_list, hyperareas, lines_formats[route + ' - ' + season], label=route + ' - ' + season, markersize=5)
        ax1.set_xlabel('Max distance from GCR [% of GCR length]')
        ax1.set_ylabel('Hyperarea []')
        ax1.legend()

# ---------------------- Aspect ratio -------------------------

    if run_aspect_ratio:

        fig1, ax1 = plt.subplots()

        for route in route_dict:
            for season in ['summer', 'winter']:

                ar_list = [0.25, 0.5, 1, 2, 4, 8, 16]

                cases = []
                for ar in ar_list:
                    cases += [(
                        set_parameters(
                            departure_time=departure_dict[season],
                            waypoints=route_dict[route],
                            aspect_ratio=ar,
                            n_side_points='from forward res'
                            ),
                        'aspect_ratio/' + route + '/' + season + '/' + str(ar)
                    )]

                solutions, ar_list = make_solutions_list(cases, ar_list)
                max_cost = max([np.max(s.costs) for s in solutions])
                max_duration = max([np.max(s.durations) for s in solutions])
                min_cost = min([np.min(s.costs) for s in solutions])
                min_duration = min([np.min(s.durations) for s in solutions])

                solutions_list = []
                colors = plt.cm.plasma(np.linspace(0, 1, len(ar_list)))

                fig, ax = plt.subplots()
                ax.set_xlabel('Voyage time [h]')
                ax.set_ylabel('Energy [kWh]')
                for i in range(len(solutions)):
                    plot_pareto(solutions[i], str(ar_list[i]), colors[i])
                    solutions_list += [(
                        solutions[i], 
                        ar_list[i], 
                        str(ar_list[i]),
                        colors[i]
                        )]
                ax.legend(title='Cell aspect ratio')

                plot_eca_curve(solutions_list, 'Cell aspect ratio')

                m, ax = plot_map(None, None)[:2]
                for i in range(len(solutions)):
                    plot_route(m, list(solutions[i].places.iloc[-1]), color=colors[i], linewidth=1, label=str(ar_list[i]))
                ax.legend(title='Cell aspect ratio')

                hyperareas = []
                for i in range(len(ar_list)):
                    hyperareas += [hyperarea(solutions[i].costs, solutions[i].durations, min_cost, min_duration, max_cost, max_duration)]
                ax1.plot(ar_list, hyperareas, lines_formats[route + ' - ' + season], label=route + ' - ' + season, markersize=5)
        ax1.set_xlabel('Cell aspect ratio []')
        ax1.set_ylabel('Hyperarea []')
        ax1.legend()

# ---------------------- Aspect ratio + adjacent points -------

    if run_aspect_ratio_and_adjacent_points:

        fig1, ax1 = plt.subplots()

        for route in ['transatlantic']:
            for season in ['winter']:

                ar_list = [0.25, 0.5, 1, 2, 4, 8, 16]
                n_adj_list = [1, 2, 3, 4, 5, 6]

                colors = plt.cm.plasma(np.linspace(0, 1, len(n_adj_list)))
                for j in range(len(n_adj_list)):
                    
                    cases = []
                    for ar in ar_list:
                        cases += [(
                            set_parameters(
                                departure_time=departure_dict[season],
                                waypoints=route_dict[route],
                                aspect_ratio=ar,
                                n_adj_points=n_adj_list[j],
                                n_side_points='from forward res'
                                ),
                            'ar_and_adj/' + route + '/' + season + '/AR' + str(ar) + '-NA' + str(n_adj_list[j])
                        )]

                    solutions, ar_list = make_solutions_list(cases, ar_list)
                    max_cost = max([np.max(s.costs) for s in solutions])
                    max_duration = max([np.max(s.durations) for s in solutions])
                    min_cost = min([np.min(s.costs) for s in solutions])
                    min_duration = min([np.min(s.durations) for s in solutions])

                    hyperareas = []
                    for i in range(len(ar_list)):
                        hyperareas += [hyperarea(solutions[i].costs, solutions[i].durations, min_cost, min_duration, max_cost, max_duration)]
                    ax1.plot(ar_list, hyperareas, '-o', color=colors[j], label=str(1+2*n_adj_list[j]), markersize=5)
        ax1.set_xlabel('Cell aspect ratio []')
        ax1.set_ylabel('Hyperarea []')
        ax1.legend(title='Number of adjacent points')

# ---------------------- Adjacent points ----------------------

    if run_adjacent_points:

        fig1, ax1 = plt.subplots()

        for route in route_dict:
            for season in ['summer', 'winter']:

                n_adj_list = [1, 2, 3, 4, 5, 6]

                cases = []
                for n_adj in n_adj_list:
                    cases += [(
                        set_parameters(
                            departure_time=departure_dict[season],
                            waypoints=route_dict[route],
                            n_adj_points=n_adj,
                            n_side_points='from forward res'
                            ),
                        'adjacent_points/' + route + '/' + season + '/' + str(n_adj)
                    )]

                solutions, n_adj_list = make_solutions_list(cases, n_adj_list)
                max_cost = max([np.max(s.costs) for s in solutions])
                max_duration = max([np.max(s.durations) for s in solutions])
                min_cost = min([np.min(s.costs) for s in solutions])
                min_duration = min([np.min(s.durations) for s in solutions])

                n_adj_list = [1+2*n for n in n_adj_list]

                solutions_list = []
                colors = plt.cm.plasma(np.linspace(0, 1, len(n_adj_list)))

                fig, ax = plt.subplots()
                ax.set_xlabel('Voyage time [h]')
                ax.set_ylabel('Energy [kWh]')
                for i in range(len(solutions)):
                    plot_pareto(solutions[i], str(n_adj_list[i]), colors[i])
                    solutions_list += [(
                        solutions[i], 
                        n_adj_list[i], 
                        str(n_adj_list[i]),
                        colors[i]
                        )]
                ax.legend(title='Number of adjacent points')

                plot_error_wrt_best_res(solutions_list, 'Number of adjacent points')

                m, ax = plot_map(None, None)[:2]
                for i in range(len(solutions)):
                    plot_route(m, list(solutions[i].places.iloc[-1]), color=colors[i], linewidth=1, label=str(n_adj_list[i]))
                ax.legend(title='Number of adjacent points')

                hyperareas = []
                for i in range(len(n_adj_list)):
                    hyperareas += [hyperarea(solutions[i].costs, solutions[i].durations, min_cost, min_duration, max_cost, max_duration)]
                ax1.plot(n_adj_list, hyperareas, lines_formats[route + ' - ' + season], label=route + ' - ' + season, markersize=5)
        ax1.set_xlabel('Number of adjacent points []')
        ax1.set_ylabel('Hyperarea []')
        ax1.legend()

# ---------------------- Time steps ---------------------------

    if run_time_resolution:

        fig1, ax1 = plt.subplots()
        texts = []

        for route in route_dict:
            for season in ['summer', 'winter']:

                n_time_steps_list = [0.3125e3,0.625e3, 1.25e3, 2.5e3, 5e3, 10e3, 20e3]

                cases = []
                for n_time_steps in n_time_steps_list:
                    cases += [(
                        set_parameters(
                            departure_time=departure_dict[season],
                            waypoints=route_dict[route],
                            n_time_steps=n_time_steps,
                            n_side_points='from forward res'
                            ),
                        'time_steps/' + route + '/' + season + '/' + str(n_time_steps)
                    )]

                solutions, n_time_steps_list = make_solutions_list(cases, n_time_steps_list)
                max_cost = max([np.max(s.costs) for s in solutions])
                max_duration = max([np.max(s.durations) for s in solutions])
                min_cost = min([np.min(s.costs) for s in solutions])
                min_duration = min([np.min(s.durations) for s in solutions])

                solutions_list = []
                colors = plt.cm.plasma(np.linspace(0, 1, len(n_time_steps_list)))

                fig, ax = plt.subplots()
                ax.set_xlabel('Voyage time [h]')
                ax.set_ylabel('Energy [kWh]')
                for i in range(len(solutions)):
                    plot_pareto(solutions[i], str(n_time_steps_list[i]), colors[i])
                    solutions_list += [(
                        solutions[i], 
                        max_duration / n_time_steps_list[i], 
                        str(n_time_steps_list[i]),
                        colors[i]
                        )]
                ax.legend(title='Number of time steps')

                plot_eca_curve(solutions_list, 'Number of time steps')

                m, ax = plot_map(None, None)[:2]
                for i in range(len(solutions)):
                    plot_route(m, list(solutions[i].places.iloc[-1]), color=colors[i], linewidth=1, label=str(n_time_steps_list[i]))
                ax.legend(title='Number of time steps')

                hyperareas = []
                for i in range(len(n_time_steps_list)):
                    hyperareas += [hyperarea(solutions[i].costs, solutions[i].durations, min_cost, min_duration, max_cost, max_duration)]
                ax1.plot(n_time_steps_list, hyperareas, lines_formats[route + ' - ' + season], label=route + ' - ' + season, markersize=5)

        ax1.set_xlabel('Number of time steps []')
        ax1.set_ylabel('Hyperarea []')
        ax1.legend()


# ---------------------- Weather time -------------------------

    if run_weather_time_resolution:

        fig1, ax1 = plt.subplots()
        texts = []

        for route in route_dict:
            for season in ['summer', 'winter']:

                weather_time_res_list = [12, 8, 6, 4, 3, 2, 1]

                cases = []
                for weather_time_res in weather_time_res_list:
                    cases += [(
                        set_parameters(
                            departure_time=departure_dict[season],
                            waypoints=route_dict[route],
                            weather_time_res_h=weather_time_res,
                            n_side_points='from forward res'
                            ),
                        'weather_time/' + route + '/' + season + '/' + str(weather_time_res)
                    )]

                solutions, weather_time_res_list = make_solutions_list(cases, weather_time_res_list)
                max_cost = max([np.max(s.costs) for s in solutions])
                max_duration = max([np.max(s.durations) for s in solutions])
                min_cost = min([np.min(s.costs) for s in solutions])
                min_duration = min([np.min(s.durations) for s in solutions])

                solutions_list = []
                colors = plt.cm.plasma(np.linspace(0, 1, len(weather_time_res_list)))

                fig, ax = plt.subplots()
                ax.set_xlabel('Voyage time [h]')
                ax.set_ylabel('Energy [kWh]')
                for i in range(len(solutions)):
                    plot_pareto(solutions[i], str(weather_time_res_list[i]), colors[i])
                    solutions_list += [(
                        solutions[i], 
                        weather_time_res_list[i], 
                        str(weather_time_res_list[i]),
                        colors[i]
                        )]
                ax.legend(title='Weather time resolution [h]')

                plot_error_wrt_best_res(solutions_list, 'Weather time resolution [h]')

                m, ax = plot_map(None, None)[:2]
                for i in range(len(solutions)):
                    plot_route(m, list(solutions[i].places.iloc[-1]), color=colors[i], linewidth=1, label=str(weather_time_res_list[i]))
                ax.legend(title='Weather time resolution [h]')

                hyperareas = []
                for i in range(len(weather_time_res_list)):
                    hyperareas += [hyperarea(solutions[i].costs, solutions[i].durations, min_cost, min_duration, max_cost, max_duration)]
                ax1.plot(weather_time_res_list, hyperareas, lines_formats[route + ' - ' + season], label=route + ' - ' + season, markersize=5)

        ax1.set_xlabel('Weather time resolution [h]')
        ax1.set_ylabel('Hyperarea []')
        ax1.legend()

# ---------------------- Weather space ------------------------

    if run_weather_space_resolution:

        fig1, ax1 = plt.subplots()
        texts = []

        for route in route_dict:
            for season in ['winter']:

                weather_space_res_list = [2, 1, 0.5, 0.25]

                cases = []
                for weather_space_res in weather_space_res_list:
                    cases += [(
                        set_parameters(
                            departure_time=departure_dict[season],
                            waypoints=route_dict[route],
                            weather_space_res_deg=weather_space_res,
                            n_side_points='from forward res'
                            ),
                        'weather_space/' + route + '/' + season + '/' + str(weather_space_res)
                    )]

                solutions, weather_space_res_list = make_solutions_list(cases, weather_space_res_list)
                max_cost = max([np.max(s.costs) for s in solutions])
                max_duration = max([np.max(s.durations) for s in solutions])
                min_cost = min([np.min(s.costs) for s in solutions])
                min_duration = min([np.min(s.durations) for s in solutions])

                solutions_list = []
                colors = plt.cm.plasma(np.linspace(0, 1, len(weather_space_res_list)))

                fig, ax = plt.subplots()
                ax.set_xlabel('Voyage time [h]')
                ax.set_ylabel('Energy [kWh]')
                for i in range(len(solutions)):
                    plot_pareto(solutions[i], str(weather_space_res_list[i]), colors[i])
                    solutions_list += [(
                        solutions[i], 
                        weather_space_res_list[i], 
                        str(weather_space_res_list[i]),
                        colors[i]
                        )]
                ax.legend(title='Weather space resolution [°]')

                plot_error_wrt_best_res(solutions_list, 'Weather space resolution [°]')

                m, ax = plot_map(None, None)[:2]
                for i in range(len(solutions)):
                    plot_route(m, list(solutions[i].places.iloc[-1]), color=colors[i], linewidth=1, label=str(weather_space_res_list[i]))
                ax.legend(title='Weather space resolution [°]')

                hyperareas = []
                for i in range(len(weather_space_res_list)):
                    hyperareas += [hyperarea(solutions[i].costs, solutions[i].durations, min_cost, min_duration, max_cost, max_duration)]
                ax1.plot(weather_space_res_list, hyperareas, lines_formats[route + ' - ' + season], label=route + ' - ' + season, markersize=5)

        ax1.set_xlabel('Weather space resolution [°]')
        ax1.set_ylabel('Hyperarea []')
        ax1.legend()


plt.show()
