from typing import Tuple, Union, Iterable, List, Callable, Optional
import os
import copy
import json

import numpy as np
import tensorflow as tf

from nnuncert.utils.clean_psi import clean_psi


def nll(y_true, y_pred):
    """Calculate negative log likelihood.

    Trains a probabilisitic neural network that predicts mean and variance for
    some input.
    """
    # negative log-likelihood criterion
    # https://arxiv.org/abs/1703.04977 eq: (8), D=1
    # https://arxiv.org/abs/1612.01474 eq: (2)
    # $s_{i}:=\log \hat{\sigma}_{i}^{2}$
    y_hat, log_var = tf.split(y_pred, num_or_size_splits=2, axis=1);
    loss = 0.5*tf.exp(-log_var)*tf.math.squared_difference(y_true, y_hat) + 0.5*log_var
    return tf.reduce_mean(loss)


class BaseModel(tf.keras.Model):
    def __init__(self,
                 net,
                 set_var_bounds: Optional[bool] = False,
                 *args, **kwargs):
        super().__init__(net.inputs, net.outputs, *args, **kwargs)
        self.net = net
        self._psi_set = False
        self._model_paras = {"set_var_bounds": set_var_bounds}

        # restrict aleatoric pred. variance to be in training range
        self.set_var_bounds = set_var_bounds

    def compile(self, *args, **kwargs):
        """Compile network, see tf.keras.Model.compile"""
        self.comp_args_kw = (args, kwargs)
        return super().compile(loss=self.myloss, *args, **kwargs)

    def recompile(self):
        """Recompile network."""
        (args, kwargs) = self.comp_args_kw
        return super().compile(loss=self.myloss, *args, **kwargs)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            path: Optional[str] = None,
            *args, **kwargs):
        """Fit network, see tf.keras.Model.fit"""
        if path is not None:
            # load model weights from path
            self.load_weights(os.path.join(*[path, "variables", "variables"])).expect_partial()
        else:
            # standard fit process
            fit_ret = super().fit(x_train, y_train, *args, **kwargs)

            # store predicted log variances for train data
            if self.pred_var is True:
                self._logvars = super().predict(x_train)[:, -1]

            return fit_ret

    def predict(self,
                x: np.ndarray,
                return_bound_info: Optional[bool] = False,
                *args, **kwargs):
        """Make prediction with NN.

        Bound aleatoric variance to stay in training range if
        self.set_var_bounds is True.
        """
        # make prediction
        pred =  super().predict(x, *args, **kwargs)

        # bound aleatoric variance to stay in training range
        if self.pred_var is True and self.set_var_bounds is True:
            bounded_lower = pred[:, -1] > self.max_logvar
            bounded_upper = pred[:, -1] < self.min_logvar
            pred[:, -1][bounded_lower] = self.max_logvar
            pred[:, -1][bounded_upper] = self.min_logvar
            if return_bound_info is True:
                return pred, (bounded_lower, bounded_upper)

        return pred

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        super().save(path, *args, **kwargs)

    def save_model_parameters(self, path: str, **kw):
        """Save model paramaters as .json to path.

        Further key-value pairs can be added with kw.
        """
        # create folder
        os.makedirs(path, exist_ok=True)

        # add kw
        d = copy.deepcopy(self._model_paras)
        d.update(kw)

        # dump settings to path
        path = os.path.join(path, "settings.json")
        with open(path, "w") as f:
            json.dump({k: d[k] for k in sorted(d)}, f, indent=2)

    @property
    def min_logvar(self):
        """Get minimum predicted log variance."""
        return min(self._logvars)

    @property
    def max_logvar(self):
        """Get maximum predicted log variance."""
        return max(self._logvars)

    @property
    def pred_var(self):
        """'True' if model predicts heteroscedastic noise."""
        return self.net.pred_var

    @property
    def myloss(self):
        """Get loss used for network.

        If network predicts heteroscedastic noise in the dataset, we use the
        NLL as loss, else MSE.
        """
        if self.pred_var is True:
            return nll
        else:
            return "mse"

    @property
    def index_last_hidden_mean(self):
        """Get index of last hidden layer."""
        return np.where([l.name.endswith("mean_-1") or l.name.endswith("output_-1") for l in self.layers])[0][0]

    @property
    def index_output(self):
        """Get index in layers of output layer."""
        return np.where([l.name.endswith("mean") or l.name.endswith("output") for l in self.layers])[0][0]

    def get_dropout_layers(self):
        """Get indices of all dropout layers."""
        idx_do =  np.where([l.name.lower().startswith("dropout")
                            for l in self.layers])[0]
        return idx_do

    def predict_psi(self,
                    x: np.ndarray,
                    clean: Optional[bool] = True,
                    add_bias: Optional[bool] = False
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict last hidden layer output values.

        Parameters
        ----------
        x : np.ndarray
            Inputs to get last layer output values.
        clean : Optional[bool] = True
            Whether to clean matrix of obtained values (psi matrix).
        add_bias : Optional[bool] = False
            Whether to add a bias row.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Cleaned psi matrix, index of removed columns during cleaning).

        """
        # make model to get hidden layer output vals
        clone = tf.keras.models.clone_model(self)
        clone.set_weights(self.get_weights())
        in_ = clone.input
        out_ = clone.layers[self.index_last_hidden_mean].output
        llmodel = tf.keras.Model(in_, out_, name="psi_pred")
        psi = llmodel.predict(x)

        # clean psi (if used for training)
        psi_idx = None
        if clean is True:
            psi, psi_idx = clean_psi(psi)
        if add_bias is True:
            psi = np.c_[np.ones(psi.shape[0]), psi]
        return psi, psi_idx
