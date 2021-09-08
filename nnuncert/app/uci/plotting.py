from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import warnings

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from nnuncert.utils.plotting import _pre_handle, _post_handle


def uci_boxplot(df: pd.DataFrame,
                fill_colors: List[str],
                width: Optional[float] = 0.6,
                ax: Optional[Axes] = None,
                title: Optional[str] = None,
                save_as: Optional[str] = None,
                **kwargs):
    """Make uci boxplot.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot.
    fill_colors : List[str]
        Colors of boxes/fliers.
    width : Optional[float]
        Width of boxpltos.
    ax : Optional[Axes]
        Axis handle, if not given new axis is created.
    title : Optional[str]
        Title of plot.
    save_as : Optional[str]
        If given, plot is saved to 'save_as'.
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    Union[Tuple[plt.Figure, plt.Axes], None]
        Return figure and axis if 'ax' is None, else None.

    """
    fig, ax, return_flag = _pre_handle(ax)

    l = len(df.columns)
    w = [width]*l
    nans = list(df.columns[df.isnull().any()])
    if len(nans) > 0:
        warnings.warn("Nan in: " + str(nans))

    bp = ax.boxplot(df.T, widths=w, patch_artist=True, showmeans=True,
                    meanline=True, **kwargs)

    # customizing
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color="black", linewidth=1)
    plt.setp(bp["medians"], linewidth=0)  # remove median
    plt.setp(bp["means"], linewidth=2.5, linestyle="-")  # set mean style

    # fill boxes and outliers by specified colors
    for i, patch in enumerate(bp['boxes']):
        patch.set(facecolor=fill_colors[i])
    for i, patch in enumerate(bp['fliers']):
        patch.set(markerfacecolor=fill_colors[i])
    ax.set_xticklabels(df.columns, rotation=90)
    plt.tight_layout()

    return _post_handle(fig, ax, return_flag, save_as, title)
