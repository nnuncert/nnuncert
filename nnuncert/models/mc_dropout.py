from typing import Tuple, Union, Iterable, List, Callable, Optional
import os

import numpy as np
import tensorflow as tf

from nnuncert.models._model_base import BaseModel
from nnuncert.models._pred_base import MCPredictions, PredConditionalGaussian
from nnuncert.utils.traintest import generate_random_split
from nnuncert.utils.io import save2txt, txt2float


class DropoutTF(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def get_dropout_layers(model) -> List[int]:
    """Get indices of all dropout layers."""
    do = np.where([l.name.lower().startswith("dropout")
                   for l in model.layers])[0]
    return do


class MCDropout(BaseModel):
    def __init__(self, net, *args, **kwargs):
        assert net.dropout_type.__name__ == "DropoutTF"
        super(MCDropout, self).__init__(net, *args, **kwargs)

    def save(self, path: str, *args, **kwargs):
        """Save model to path."""
        # save std with keras
        super().save(path, *args, **kwargs)

        # create folder for best hyper and save bet dropout rate
        path_hp = os.path.join(path, "hyper")
        os.makedirs(path_hp, exist_ok=True)
        file = os.path.join(*[path_hp, "dropout.txt"])
        save2txt(self.best_dropout, file)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            normalize_y: Optional[bool] = True,
            grid_dropout: Optional[List[float]] = [0.005, 0.01, 0.05, 0.1],
            path: Optional[str] = None,
            val_ratio: Optional[float] = 0.2,
            conv_factor: Optional[int] = 1,
            *args, **kw):
        """Fit MC dropout to data.

        Finds best dropout rate in 'grid_dropout' on a validation set with ratio
        'val_ratio'. With best dropout rate, model is fitted to full data.

        Parameters
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        normalize_y : Optional[bool] = True
            Whether to standardize targets for fit.
        grid_dropout : Optional[List[float]] = [0.005, 0.01, 0.05, 0.1]
            Dropout rates to perform grid search on.
        path : Optional[str] = None
            If 'path' is given, model will be loaded from 'path'.
        val_ratio : Optional[float] = 0.2
            Ratio of validation set for finding best dropout rate.
        conv_factor : Optional[int] = 1
            Fit dropout after grid search with 'conv_factor' * 'epochs'.

        """
        self._model_paras["grid_dropout"] = grid_dropout
        self._model_paras["val_ratio"] = val_ratio
        self._model_paras["conv_factor"] = conv_factor

        # standardize y_train to have zero mean and unit variance
        if normalize_y is True:
            self.y_train_std = y_train.std()
            self.y_train_mean = y_train.mean()
            y_train = (y_train - self.y_train_mean) / self.y_train_std
        else:
            # save for rescaling of predictions
            self.y_train_std = 1
            self.y_train_mean = 0

        if path is not None:  # reload model...
            # load stored weights and best dropout rate
            self.load_weights(os.path.join(*[path, "variables",
                                             "variables"])).expect_partial()

            # load dropout rate and set in all layers
            self.best_dropout = txt2float(os.path.join(*[path, "hyper",
                                                         "dropout.txt"]))
            for i in get_dropout_layers(self):
                self.layers[i].rate = self.best_dropout

        else:
            # store initial weights as not to refit an existing model
            initial_weights = self.get_weights()

            if len(grid_dropout) == 1:
                self.best_dropout = grid_dropout[0]

            # find best dropout rate in grid_dropout
            else:
                # calculate loss for all dropout rates in grid
                losses = []
                for do in grid_dropout:
                    # reset dropout rate and recompile
                    for i in get_dropout_layers(self):
                        self.layers[i].rate = do

                    # reset fitted weights due to grid iteration
                    self.set_weights(initial_weights)
                    self.recompile()

                    # fit network for new droprate on subset of train data
                    hist = super().fit(x_train, y_train, validation_split=val_ratio, *args, **kw)

                    # get loss for validation data
                    losses.append(hist.history["val_loss"][-1])

                # store best dropout rate
                self.best_dropout = grid_dropout[np.argmin(losses)]
                
            self._model_paras["best_dropout"] = self.best_dropout

            # change dropout rate to best found in grid
            for i in get_dropout_layers(self):
                self.layers[i].rate = self.best_dropout

            # reset fitted weights due to grid iteration
            self.set_weights(initial_weights)
            self.recompile()

            # retrain model on best dropout rate, conv_factor times epochs
            kw["epochs"] *= conv_factor
            super().fit(x_train, y_train, *args, **kw)

    def make_prediction(self, x, *args, **kwargs):
        """Get prediction object for x."""
        return MCDropoutPred(self, x, *args, **kwargs)


class MCDropoutPred(PredConditionalGaussian):
    def __init__(self,
                 model: MCDropout,
                 x: np.ndarray,
                 npreds: Optional[int] = 1000,
                 *args, **kwargs):
        super(MCDropoutPred, self).__init__(x.shape[0], *args, **kwargs)
        
        # store number of predictions in model params, for interpretability of results
        model._model_paras["npreds"] = npreds

        # make npred predictions for x
        x_pred = np.repeat(x, npreds, axis=0)
        
        # get mean predictions only (homosceastic version)
        if model.pred_var is False:
            y_pred = model.predict(x_pred).T.reshape((-1, npreds))
            y_var = np.zeros(x.shape[0]).ravel()
        
        # get mean and variance (heteroscedastic version)
        else:
            y_pred, self.y_log_var = model.predict(x_pred).T.reshape((2, -1, npreds))
            y_var = np.exp(self.y_log_var.mean(axis=1))

        # rescale predictive mean and variance
        self.y_pred = MCPredictions(y_pred)*model.y_train_std + model.y_train_mean
        self.y_var = y_var*model.y_train_std**2

        # make Gaussians for each predictive density
        self._make_gaussians()

    @property
    def var_aleatoric(self):
        # https://arxiv.org/abs/1703.04977 eq: (9), part 2
        return self.y_var

    @property
    def var_epistemic(self):
        # https://arxiv.org/abs/1703.04977 eq: (9), part 1
        return self.y_pred.var(axis=1).np

    @property
    def pred_mean(self):
        return self.y_pred.mean(axis=1).np
