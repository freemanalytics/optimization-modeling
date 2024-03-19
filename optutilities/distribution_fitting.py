import numpy as np


def plot_histogram_for_single_array(
    array: np.ndarray,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.histplot(
        array,
        stat='probability',
        label='Array 1',
    )
    ax.spines[['right', 'top']].set_visible(False)

    plt.show()


def plot_histogram_for_two_arrays(
    array1: np.ndarray,
    array2: np.ndarray,
    array1_label: str = 'Array 1',
    array2_label: str = 'Array 2',
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.histplot(
        array1,
        stat='probability',
        label=array1_label,
    )

    sns.histplot(
        array2,
        stat='probability',
        label=array2_label,
    )
    ax.legend()
    ax.spines[['right', 'top']].set_visible(False)

    plt.show()


def get_best_parametric_distribution(
    observation_array: np.ndarray,
    fit_all: bool = False,
    plot_fit: bool = False,
):
    import warnings
    warnings.simplefilter('ignore')
    import scipy.stats as st
    from tqdm.auto import tqdm

    min_val = np.floor(observation_array.min()).astype(int)
    max_val = np.ceil(observation_array.max()).astype(int)
    bins = [i for i in range(min_val, max_val+1, 1)]

    y, x = np.histogram(
        observation_array,
        bins=bins,
        density=True,
    )

    scipy_stats_objects = dir(st)
    if fit_all:
        distribution_choices = [
            item for item in scipy_stats_objects
            if 'pdf' in eval(f'dir(st.{item})')
            ]
    else:
        distribution_choices = [
            st.beta,
            st.expon,
            st.gamma,
            st.lognorm,
            st.norm,
            st.triang,
            st.truncnorm,
            st.uniform,
        ]

    best_fit_name = None
    best_fit_rsse = np.inf
    best_fit_dist = None
    n = len(observation_array)

    for current_dist in tqdm(distribution_choices):
        current_dist_best_params = current_dist.fit(observation_array)
        current_dist_best_fit = current_dist(*current_dist_best_params)
        current_dist_observations = current_dist_best_fit.rvs(n) # noqa

        comparison_y, _ = np.histogram(
            current_dist_observations,
            bins=bins,
            density=True,
        )
        rsse = np.sum((comparison_y - y)**2)
        if rsse < best_fit_rsse:
            best_fit_rsse = rsse
            best_fit_name = current_dist.name
            best_fit_dist = current_dist_best_fit

    print(f'{best_fit_name = }') # noqa
    if plot_fit:
        best_fit_observations = best_fit_dist.rvs(n)

        plot_histogram_for_two_arrays(
            array1=observation_array,
            array2=best_fit_observations,
            array1_label='Sample',
            array2_label='Fit',
            )

    return best_fit_dist
