from typing import Tuple

import numpy as np
from numpy.linalg import matrix_rank


def clean_psi(psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove zero cols and dependent columns from matrix.

    Parameters
    ----------
    psi : np.ndarray
        Basis function matrix from hidden layer output.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (Cleanded matrix, index of deleted columns).

    """
    psi2 = psi.copy()

    # remove zero cols
    idx_zero = np.where(psi2.sum(axis=0) == 0)[0]
    psi2 = psi2[:, ~(psi2.sum(axis=0) == 0)]

    # remove linearly dependent columns
    if matrix_rank(psi2) < psi2.shape[1]:
        psi_tmp = psi2[:, 0].reshape(-1, 1)
        psi_out = psi_tmp.copy()
        idx = [0]

        # find columns that increase rank of matrix
        for i in range(psi2.shape[1] - 1):
            # get proper index
            j = i + 1

            # add to tmp and check matrix rank
            psi_tmp = np.c_[psi_tmp, psi2[:, j].reshape(-1, 1)]
            if matrix_rank(psi_tmp) > matrix_rank(psi_out):
                psi_out = np.c_[psi_out, psi2[:, j].reshape(-1, 1)]
                idx.append(j)

        idx = np.array(idx)

    else:
        psi_out = psi2
        idx = np.array(range(psi2.shape[1]))

    for z in idx_zero:
        idx[idx >= z] += 1

    return psi_out, idx
