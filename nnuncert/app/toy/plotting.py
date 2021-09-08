from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nnuncert.utils.plotting import _pre_handle, _post_handle


def make_toy_plot(pred,
                  x_train: np.ndarray,
                  y_train: np.ndarray,
                  x0: np.ndarray,
                  reg_func: Callable,
                  colors: [str, str, str],
                  std_devs: int = 2,
                  ax: Union[None, plt.Axes] = None,
                  save_as: Union[None, str] = None,
                  title: Union[None, str] = None
                  ) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    """Make 1D toy plot of pred. mean and uncertainty.

    Parameters
    ----------
    pred :
        Prediction object.
    x_train : np.ndarray
    y_train : np.ndarray
    x0 : np.ndarray
        Linspace to cover x-range of plot.
    reg_func : Callable
        Function to get ground truth.
    std_devs : int
       Controls How many standard deviations of uncertainty to plot above/below
       mean.
    colors : [str, str, str]
        Colors for [scatter, predictive mean, predictive variance].
    ax : Union[None, plt.Axes]
    save_as : Union[None, str]
    title : Union[None, str]

    Returns
    -------
    Union[Tuple[plt.Figure, plt.Axes], None]
        Fig/ax if no ax is given, else None.

    """
    fig, ax, return_flag = _pre_handle(ax)

    # ground truth and training data
    ax.plot(x0, reg_func(x0), "--", color="black", label="True")
    ax.scatter(x_train, y_train, label="training Data", color=colors[0])

    # predictive mean +- 'std' standard deviations
    ax.plot(x0, pred.pred_mean, label="mean prediction", color=colors[1])
    lower = pred.pred_mean - std_devs*pred.std_total
    upper = pred.pred_mean + std_devs*pred.std_total
    ax.fill_between(x0, lower, upper, alpha=.2, color=colors[2])

    return _post_handle(fig, ax, return_flag, save_as, title)


def contour_plot_2d(x1: np.ndarray,
                    x2: np.ndarray,
                    z: np.ndarray,
                    x_train = None,
                    ax: Union[None, plt.Axes] = None,
                    fig_ = None,
                    make_colbar: bool = True,
                    title: str = "$\sigma[f(\mathbf{x})]$",
                    save_as: Union[None, str] = None,
                    **kwargs
                    ) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    """Create 2D contour plot.

    Parameters
    ----------
    x1 : np.ndarray
        Integer index, i.e. x1 = range(M), where M is # columns in z.
    x2 : np.ndarray
        Integer index, i.e. x2 = range(N), where N is # columns in z.
    z : np.ndarray
        The height values over which the contour is drawn.
    x_train : Union[None, np.ndarray] (optional)
        Training data to draw (noisy) samples
    ax : Union[None, plt.Axes] (optional)
        Plot to ax.
    title : str
    save_as : Union[None, str]
    **kwargs : type
        **kwargs mapped to 'ax.contourf(x1, x2, z, **kwargs)'.

    Returns
    -------
    Union[Tuple[plt.Figure, plt.Axes], None]
        Fig/ax if no ax is given, else None.

    """
    fig, ax, return_flag = _pre_handle(ax)
    if fig_ is not None:
        fig = fig_

    # plot contour to ax
    levels = kwargs.setdefault('levels', 200)
    cnt = ax.contourf(x1, x2, z, **kwargs)
    for c in cnt.collections:
        c.set_edgecolor("face")

    # add colorbar
    if make_colbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(cnt, cax=cax, ax=ax, format='%0.2f')
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

    # add scatter of sample data if given
    if x_train is not None:
        x_train, c = x_train
        ax.scatter(x_train[:, 0], x_train[:, 1], marker='+', color=c)

    # annotate plot
    ax.set_aspect('equal')
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')

    return _post_handle(fig, ax, return_flag, save_as, title)
