from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import os
import json

import numpy as np

from nnuncert.models._network import MakeNet
from nnuncert.models._pred_base import PredConditionalGaussian
from nnuncert.models.pbp._pbp import load_PBP_net_from_file, PBP_net


class PBPModel():
    def __init__(self, net: MakeNet, *args, **kwargs):
        self.n_hidden_units = net.hidden_units

    def compile(self, *args, **kwargs):
        """Dummy funcion for consistency."""
        pass

    def save(self, path: str):
        """Save model to path."""
        self.pbp.save_to_file(os.path.join(path, "pbp"))

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            verbose: Optional[int] = 0,
            path: Optional[str] = None,
            *args, **kwargs):
        """Fit PBP to data.

        Parameters
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        epochs : int
            Number of epochs for fitting.
        verbose : Optional[int]
        path : Optional[str]
            If 'path' is given, model will be loaded from 'path'.

        """
        if path is not None:
            self.pbp = load_PBP_net_from_file(os.path.join(path, "pbp"))
        else:
            self.pbp = PBP_net(x_train, y_train, self.n_hidden_units, n_epochs=epochs, normalize=False)

    def make_prediction(self, x: np.ndarray):
        """Get prediction object for x."""
        return PBPPred(self, x)

    def save_model_parameters(self, path: str, **kw):
        """Save model paramaters as .json to path.

        Further key-value pairs can be added with kw.
        """
        # create folder
        os.makedirs(path, exist_ok=True)

        # add kw
        d = {"layer_sizes": self.n_hidden_units}
        d.update(kw)

        # dump settings to path
        path = os.path.join(path, "settings.json")
        with open(path, "w") as f:
            json.dump({k: d[k] for k in sorted(d)}, f, indent=2)


class PBPPred(PredConditionalGaussian):
    def __init__(self, pbpmodel: PBPModel, x: np.ndarray):
        self.xlen = x.shape[0]
        self.mean, self.var, self.var_noise = pbpmodel.pbp.predict(x)
        self._make_gaussians()

    @property
    def pred_mean(self):
        """Get predictive mean."""
        return self.mean

    @property
    def var_aleatoric(self):
        """Get aleatoric variance."""
        return self.var_noise

    @property
    def var_epistemic(self):
        """Get epistemic variance."""
        return self.var
