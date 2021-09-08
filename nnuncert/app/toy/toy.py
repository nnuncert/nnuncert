from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import pandas as pd
import numpy as np


def make_toy_data(x: np.ndarray, reg_func: Callable,
                  sigma: Union[int, float, Callable],
                  seed: Union[None, int, np.random.Generator] = None
                  ) -> pd.DataFrame:
    """Create toy data from feature values 'x' and a regression function.

    Parameters
    ----------
    x : np.ndarray
        Feature array, must agree in dimension with 'reg_func' inputs.
    reg_func : Callable
        Function to create response from 'x'.
    sigma : Union[int, float, Callable]
        Noise value or noise function.
    seed : Union[None, int, np.random.Generator]
        Random generator input.

    Returns
    -------
    pd.DataFrame
        Toy data with features 'x_1', 'x_2', ... and (noisy) response 'y'.

    """
    rng = np.random.default_rng(seed)

    # create dataframe from array and handle naming
    df = pd.DataFrame(x)
    df.columns = ["x" + str(i + 1) for i in range(df.shape[1])]

    # make noise
    if isinstance(sigma, (int, float)):
        noise = rng.normal(0, sigma, len(df))
    else:
        noise = sigma(x)

    # add noise to truth and set response column 'y' in dataframe
    df["y"] = reg_func(x).ravel() + noise
    return df


def sample_from_circle(r: Union[int, float], var: Union[int, float] = 0.2,
                       mean: Union[np.ndarray, Tuple, List] = [0, 0],
                       n: int = 100,
                       seed: Union[None, int, np.random.Generator] = None) -> np.ndarray:
    """Generate sample points around circle with radius 'r'."

    Parameters
    ----------
    r : Union[int, float]
        Radius of sample circle.
    var : Union[int, float]
        Variance of radius for sampling.
    mean : Union[np.ndarray, Tuple, List]
        Specify mid-point of circle.
    n : int
        Number of samples.
    seed : Union[None, int, np.random.Generator]
        Random generator input.

    Returns
    -------
    np.ndarray
        Feature samples.

    """
    rng = np.random.default_rng(seed)

    # generate random degrees
    t = rng.random(n) * 360 - 180

    # generate random radius
    r = r + rng.random(n) * var

    # get x/y values
    x1 = mean[0] + (r * np.cos(t)).reshape(-1, 1)
    x2 = mean[1] + (r * np.sin(t)).reshape(-1, 1)

    return np.hstack((x1, x2))


def gen_2d_gaussian_samples(mu: list, var: Union[int, float] = 0.01,
                            ppc: int = 100,
                            seed: Union[None, int, np.random.Generator] = None) -> np.ndarray:
    """Create clusters of 2D gaussian samples.

    Parameters
    ----------
    mu : list
        List of (mu_0, mu_1) for mean points to sample cluster.
    var : Union[int, float]
        Variance of sampling.
    ppc : int
        Points per cluster.
    seed : Union[None, int, np.random.Generator]
        Random generator input.

    Returns
    -------
    np.ndarray
        Feature samples.

    """
    rng = np.random.default_rng(seed)

    # make covariance matrix
    s = np.eye(2)*var

    # init x data and fill around clusters specified by mu
    x = np.empty((0, 2))
    for m in mu:
        # generate new cluster and add to data
        clus = rng.multivariate_normal(m, s, ppc)
        x = np.vstack((x, clus))

    return x


def input2grid(x_range, y_range, ppa: int
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate grid features with np.meshgrid.

    Parameters
    ----------
    x_range
        (left bound, right bound) of x range.
    y_range
        (left bound, right bound) of y range.
    ppa : int
        Number of points per axis.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (Grid as (n x 2), x_grid, y_grid).

    """
    x = np.linspace(*x_range, ppa)
    y = np.linspace(*y_range, ppa)
    xx, yy = np.meshgrid(x, y)
    grid_arr = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    return grid_arr, xx, yy
