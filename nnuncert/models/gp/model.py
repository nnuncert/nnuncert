from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import numpy as np
import gpflow

from nnuncert.models._network import MakeNet
from nnuncert.models.gp.kernel import ReLUKernel


class GPModel:
    def __init__(self, net: MakeNet):
        self.num_layers = len(net.hidden_units)

    def compile(self, *args, **kwargs):
        """Dummy function for consistency."""
        pass

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            sigma_w: float,
            sigma_b: float,
            noise_variance: float,
            *args, **kwargs):
        """Fit GP to data.

        Parameters
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        sigma_w : float
            Prior for weights variance.
        sigma_b : float
            Prior for bias variance.
        noise_variance : float
            Variance of aleatoric noise in data

        """
        self.ker = ReLUKernel(prior_weight_std=sigma_w,
                              prior_bias_std=sigma_b,
                              depth=self.num_layers)
        self.gp = gpflow.models.GPR(data=(x_train, y_train.reshape(x_train.shape[0], -1)),
                                    kernel=self.ker,
                                    noise_variance=noise_variance)

    def make_prediction(self, x: np.ndarray):
        """Get prediction object for x."""
        return GPPred(self, x)


class GPPred:
    def __init__(self, gpmodel: GPModel, x: np.ndarray):
        self.pred_mean, self.var = gpmodel.gp.predict_y(x)
        self.pred_mean = self.pred_mean.numpy().ravel()
        self.var = self.var.numpy().ravel()
        self.std_total = self.var**0.5
