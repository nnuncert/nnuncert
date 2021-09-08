from typing import Tuple


def index_to_rowcol(index: int, width: int) -> Tuple[int, int]:
    """Translate 1D index to 2D index.

    Parameters
    ----------
    index : int
    width : int
        Width of target 2D matrix.

    Returns
    -------
    Tuple[int, int]
        Row / column indices for target 2D matrix.

    """
    row = (int)(index / width)
    col = index % width
    return row, col
