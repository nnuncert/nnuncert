from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import matplotlib.pyplot as plt


def _pre_handle(ax: Union[plt.Axes, None]
                ) -> Tuple[Union[plt.Figure, None], plt.Axes, bool]:
    """Pre handle for plots to either add to existing or create new figure.

    If axis does not exist, create figure and axis.

    Parameters
    ----------
    ax : Union[plt.Axes, None]

    Returns
    -------
    Tuple[Union[plt.Figure, None], plt.Axes, bool]
        Figure (if ax is None), axis (created if ax is None), return_flag.

    """
    return_flag = False
    fig = None
    if ax is None:
        # make fig, axis if axis does not exist
        return_flag = True
        fig, ax = plt.subplots()

    return fig, ax, return_flag


def _post_handle(fig: plt.Figure, ax: plt.Axes, return_flag: bool,
                 save_as: Union[str, None], title:  Union[str, None]
                 ) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    """Post handle for plotting

    Sets title for ax, saves plot, handles return.

    Parameters
    ----------
    fig : plt.Figure
    ax : plt.Axes
    return_flag : bool
    save_as : Union[str, None]
        Save plot to 'save_as' if not None.
    title : Union[str, None]
        Set title of axis to 'title' if not None.

    Returns
    -------
    Union[Tuple[plt.Figure, plt.Axes], None]
        Return figure and axis if 'return_flag' == True

    """
    if title is not None:
        ax.set_title(title)
    if isinstance(save_as, str):
        fig.savefig(save_as)
    if return_flag is True:
        return fig, ax
