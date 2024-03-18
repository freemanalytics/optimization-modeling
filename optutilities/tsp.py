import gurobipy
import pandas as pd


def generate_random_tsp_data(
    n_locations: int = 10,
    random_seed: int = 42,
) -> dict:
    '''
    Generates a random set of locations and distances between locations for use
    as input data for a traveling salesman problem.
    '''

    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import euclidean_distances

    locations = [i for i in range(1, n_locations+1)]

    np.random.seed(random_seed)

    coordinates = pd.DataFrame(
        np.random.uniform(low=0, high=100, size=(n_locations, 2)),
        index=locations,
        columns=['x', 'y'],
    )

    location_distances = pd.DataFrame(
        euclidean_distances(coordinates, coordinates),
        index=locations,
        columns=locations,
    ).reset_index().rename(
        columns={'index': 'start'}
    ).melt(
        id_vars=['start']
    ).rename(
        columns={
            'variable': 'end',
            'value': 'distance',
        }
    ).set_index(
        ['start', 'end']
    )['distance'].to_dict()

    return {
        'coordinates_df': coordinates,
        'locations': locations,
        'location_distances': location_distances,
    }


def generate_demo_tsp_data() -> dict:
    '''
    Generates a small set of locations and distances between locations for use
    as input data for traveling salesman problem demonstrations.
    '''

    import pandas as pd
    from sklearn.metrics.pairwise import euclidean_distances

    locations = [1, 2, 3, 4, 5, 6]

    coordinates_list = [
        [0, 1],
        [1, 2],
        [2, 2],
        [3, 1],
        [2, 0],
        [1, 0],
    ]

    coordinates = pd.DataFrame(
        coordinates_list,
        index=locations,
        columns=['x', 'y'],
    )

    location_distances = pd.DataFrame(
        euclidean_distances(coordinates, coordinates),
        index=locations,
        columns=locations,
    ).reset_index().rename(
        columns={'index': 'start'}
    ).melt(
        id_vars=['start']
    ).rename(
        columns={
            'variable': 'end',
            'value': 'distance',
        }
    ).set_index(
        ['start', 'end']
    )['distance'].to_dict()

    return {
        'coordinates_df': coordinates,
        'locations': locations,
        'location_distances': location_distances,
    }


def generate_tsp_plot(
    coordinates_df: pd.DataFrame,
    tour_list: list = [],
    label_locations: bool = True,
    figsize_tuple: tuple = (5, 5),
) -> None:
    '''
    Generates plot for TSP instance.

    Arguments
     - coordinates_df - pandas DataFrame containing location coordinates
     - tour_list - a list specifying a tour covering the locations
     - label_locations - boolean defining whether or not to label locations
    '''

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize_tuple)

    coordinates_df.plot(
        kind='scatter',
        x='x',
        y='y',
        edgecolor='k',
        ax=ax,
    )

    ax.spines[['right', 'top']].set_visible(False)

    if label_locations:
        coordinate_dicts = coordinates_df.to_dict(orient='index')
        for customer, coordinate_dict in coordinate_dicts.items():
            ax.annotate(
                customer,
                (coordinate_dict['x']*1.02, coordinate_dict['y']*1.02),
                )

    if tour_list:
        coordinate_dicts = coordinates_df.to_dict(orient='index')
        x_list, y_list = [], []
        for location_idx, location in enumerate(tour_list[:-1], 0):
            x_list.append([
                coordinate_dicts[location]['x'],
                coordinate_dicts[tour_list[location_idx+1]]['x'],
                ])
            y_list.append([
                coordinate_dicts[location]['y'],
                coordinate_dicts[tour_list[location_idx+1]]['y'],
                ])
        x_list.append([
            coordinate_dicts[tour_list[-1]]['x'],
            coordinate_dicts[tour_list[0]]['x'],
            ])
        y_list.append([
            coordinate_dicts[tour_list[-1]]['y'],
            coordinate_dicts[tour_list[0]]['y'],
            ])

        ax.plot(
            x_list,
            y_list,
        )

    plt.show()


def subtour_elimination(
    gurobi_model,
    where,
) -> None:

    if where == gurobipy.GRB.callback.MIPSOL:

        locations = gurobi_model._locations
        vals = gurobi_model.cbGetSolution(gurobi_model._vars)

        unvisited = set(locations)
        tours = []
        while unvisited:
            origin = unvisited.pop()
            current_tour = []
            current_tour.append(origin)
            continue_tour = True
            while continue_tour:
                continue_tour = False
                for destination in locations:
                    if vals[origin, destination] > 0.1:
                        current_tour.append(destination)
                        if destination in unvisited:
                            unvisited.remove(destination)
                            origin = destination
                            continue_tour = True
            tours.append(current_tour)

        for tour in tours:
            if len(set(tour)) < len(locations):
                X_sum = gurobipy.LinExpr()
                for stop_idx, stop in enumerate(tour[:-1]):
                    X_sum.add(gurobi_model._vars[stop, tour[stop_idx+1]])
                gurobi_model.cbLazy(X_sum <= len(set(tour)) - 1)


def solve_tsp(
    locations: list,
    distance_dict: dict,
    warmstarts: list = [],
    time_limit_seconds: int = 60,
    verbose: bool = True,
) -> dict:

    model = gurobipy.Model('TSP')

    X = model.addVars(
        locations,
        locations,
        vtype=gurobipy.GRB.BINARY,
        name='X',
    )
    model.setParam('TimeLimit', time_limit_seconds)
    if not verbose:
        model.setParam('OutputFlag', 0)

    total_cost = gurobipy.LinExpr()
    for i in locations:
        for j in locations:
            total_cost.add(distance_dict[i, j]*X[i, j])

    model.setObjective(total_cost, sense=gurobipy.GRB.MINIMIZE)

    for i in locations:
        j_sum = gurobipy.LinExpr()
        for j in locations:
            if i != j:
                j_sum.add(X[i, j])
        model.addConstr(j_sum == 1)

    for j in locations:
        i_sum = gurobipy.LinExpr()
        for i in locations:
            if i != j:
                i_sum.add(X[i, j])
        model.addConstr(i_sum == 1)

    for i in locations:
        model.addConstr(X[i, i] == 0)

    if warmstarts:
        if isinstance(warmstarts[0], list):
            model.NumStart = len(warmstarts)
            model.update()

            for warmstart_index, warmstart in enumerate(warmstarts):
                model.params.StartNumber = warmstart_index

                for stop_idx, stop in enumerate(warmstart[:-1]):
                    X[stop, warmstart[stop_idx+1]].start = 1
                X[warmstart[-1], warmstart[0]].start = 1

        else:
            for stop_idx, stop in enumerate(warmstarts[:-1]):
                X[stop, warmstarts[stop_idx+1]].start = 1
            X[warmstarts[-1], warmstarts[0]].start = 1

    model._vars = X
    model._locations = locations

    model.setParam('LazyConstraints', 1)
    # you can also use model.params.LazyConstraints = 1

    model.optimize(subtour_elimination)

    return {
        'model': model,
        'X': X,
    }


def convert_tsp_X_to_tour(
    locations: list,
    X: gurobipy.tupledict,
) -> list:

    origin = locations[0]
    tour = [origin]
    while len(tour) < len(locations):
        for destination in locations:
            if X[origin, destination].x > 0.1:
                tour.append(destination)
                origin = destination
                break

    return tour


def get_tour_distance(
    tour_list: list,
    distance_dict: dict,
) -> float:

    distance = 0
    for stop_idx, stop in enumerate(tour_list[:-1]):
        distance += distance_dict[stop, tour_list[stop_idx+1]]
    distance += distance_dict[tour_list[-1], tour_list[0]]

    return distance


def get_nearest_neighbor_tour(
    locations_list: list,
    origin: int,
    distance_dict: dict,
) -> float:
    import numpy as np

    NN_sequence = [origin]
    unvisited = set(locations_list) - set(NN_sequence)

    while unvisited:
        closest_neighbor = None
        minimum_distance = np.inf

        for potential_destination in unvisited:
            if distance_dict[origin, potential_destination] < minimum_distance:
                closest_neighbor = potential_destination
                minimum_distance = distance_dict[origin, potential_destination]

        NN_sequence.append(closest_neighbor)
        unvisited = set(locations_list) - set(NN_sequence)
        origin = closest_neighbor

    return NN_sequence
