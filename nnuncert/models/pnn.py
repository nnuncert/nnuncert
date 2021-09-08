from typing import Tuple, Union, Iterable, List, Callable, Optional

import numpy as np

from nnuncert.models._model_base import BaseModel
from nnuncert.models._pred_base import PredConditionalGaussian


class PNN(BaseModel):
    def __init__(self, net, *args, **kwargs):
        assert net.pred_var is True
        super(PNN, self).__init__(net, *args, **kwargs)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            normalize_y: Optional[bool] = True,
            *args, **kw):
        """Fit PNN to data.

        Parameters
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        normalize_y : Optional[bool] = True
            Whether to standardize targets for fit.

        """
        if normalize_y is True:
            self.y_train_std = y_train.std()
            self.y_train_mean = y_train.mean()
            y_train = (y_train - self.y_train_mean) / self.y_train_std
        else:
            self.y_train_std = 1
            self.y_train_mean = 0
        super().fit(x_train, y_train, *args, **kw)

    def make_prediction(self, x):
        """Get prediction object for x."""
        return PNNPred(self, x)


class PNNPred(PredConditionalGaussian):
    def __init__(self, model: PNN, x: np.ndarray):
        # assert isinstance(model, PNN)

        super(PNNPred, self).__init__(x.shape[0])
        assert model.pred_var is True

        self.mean, y_log_var = model.predict(x).T
        self.mean = self.mean*model.y_train_std + model.y_train_mean
        self.y_var = np.exp(y_log_var)*model.y_train_std**2
        self._make_gaussians()

    @property
    def var_aleatoric(self):
        return self.y_var

    @property
    def pred_mean(self):
        return self.mean
