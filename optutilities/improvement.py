def apply_adjacent_pairwise_interchange(
    input_list: list,
) -> list:
    import numpy as np

    a = np.random.randint(0, len(input_list)-1)

    mutated = list(input_list)
    mutated[a], mutated[a + 1] = mutated[a + 1], mutated[a]

    return mutated


def apply_pairwise_interchange(
    input_list: list,
) -> list:
    import numpy as np

    a, b = np.random.choice(
        range(0, len(input_list)),
        size=2,
        replace=False,
    )

    mutated = list(input_list)
    mutated[a], mutated[b] = mutated[b], mutated[a]

    return mutated


def apply_subsequence_reversal(
    input_list: list,
) -> list:
    import numpy as np

    a, b = np.random.choice(
        range(0, len(input_list)+1),
        size=2,
        replace=False,
    )
    a, b = sorted([a, b])

    mutated = list(input_list)
    mutated[a:b] = mutated[a:b][::-1]

    return mutated


def apply_three_opt(
    input_list: list,
):
    """
    This function is adapted from that found at:
    https://stackoverflow.com/questions/21205261/3-opt-local-search-for-tsp

    Original description: In the broad sense, 3-opt means choosing any three
    edges ab, cd and ef and chopping them, and then reconnecting (such that the
    result is still a complete tour). There are eight ways of doing
    it. One is the identity, 3 are 2-opt moves (because either ab, cd,
    or ef is reconnected), and 4 are 3-opt moves (in the narrower
    sense).
    """
    import random

    n = len(input_list)
    mutated = list(input_list)

    # choose 3 unique edges defined by their first node
    a, c, e = random.sample(range(n+1), 3)

    # without loss of generality, sort
    a, c, e = sorted([a, c, e])
    b, d, f = a+1, c+1, e+1

    # in the following slices, the nodes abcdef are referred to by
    # name. x:y:-1 means step backwards. anything like c+1 or d-1
    # refers to c or d, but to include the item itself, we use the +1
    # or -1 in the slice
    cases = {
        'identity': mutated[:a+1] + mutated[b:c+1] + mutated[d:e+1] + mutated[f:], # noqa
        '2-opt-1': mutated[:a+1] + mutated[b:c+1] + mutated[e:d-1:-1] + mutated[f:], # noqa
        '2-opt-2': mutated[:a+1] + mutated[c:b-1:-1] + mutated[d:e+1] + mutated[f:], # noqa
        '3-opt-1': mutated[:a+1] + mutated[c:b-1:-1] + mutated[e:d-1:-1] + mutated[f:], # noqa
        '3-opt-2': mutated[:a+1] + mutated[d:e+1] + mutated[b:c+1] + mutated[f:], # noqa
        '3-opt-3': mutated[:a+1] + mutated[d:e+1] + mutated[c:b-1:-1] + mutated[f:], # noqa
        '3-opt-4': mutated[:a+1] + mutated[e:d-1:-1] + mutated[b:c+1] + mutated[f:], # noqa
        '2-opt-3': mutated[:a+1] + mutated[e:d-1:-1] + mutated[c:b-1:-1] + mutated[f:], # noqa
    }

    case = random.choice(list(cases.keys()))

    return cases[case]


def get_improvement_plot(
    progress_dict: dict,
    figsize_tuple: tuple = (6, 4),
    logx: bool = True,
    title: str = '',
) -> None:

    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots(1, 1, figsize=figsize_tuple)

    pd.DataFrame.from_dict(
        progress_dict,
        orient='index',
        columns=['value']
    ).reset_index().rename(
        columns={'index': 'iteration'}
    ).plot(
        x='iteration',
        y='value',
        logx=logx,
        ax=ax,
        legend=False,
    )

    ax.spines[['right', 'top']].set_visible(False)
    ax.axhline(
        min(progress_dict.values()),
        color='k',
        linestyle='--',
    )

    if title:
        ax.set_title(title)
