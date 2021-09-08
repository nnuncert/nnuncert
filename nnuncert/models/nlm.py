from typing import Tuple, Union, Iterable, List, Callable, Optional
import itertools
import os

import numpy as np
from scipy.sparse import dia_matrix

from nnuncert.models._network import MakeNet
from nnuncert.models._model_base import BaseModel
from nnuncert.models._pred_base import PredConditionalGaussian
from nnuncert.utils.traintest import generate_random_split
from nnuncert.utils.io import save2txt, txt2float


TAU2_GRID = list(itertools.chain(*[(0.5*10**x, 10**x) for x in range(-2, 4)]))


class NLM(BaseModel):
    def __init__(self,
                 net: MakeNet,
                 account_for_bias: Optional[bool] = False,
                 *args, **kwargs):
        assert net.pred_var is True
        super(NLM, self).__init__(net, *args, **kwargs)
        self.account_for_bias = account_for_bias

    def predict_psi(self, x: np.ndarray):
        """Predict hidden layer outputs for 'x'."""
        return super().predict_psi(x, clean=False, add_bias=self.account_for_bias)[0]

    def save(self, path: str, *args, **kwargs):
        """Save model to path."""
        super().save(path, *args, **kwargs)

        # save best value for tau^2 (controls prior variance for betas)
        path_hp = os.path.join(path, "hyper")
        os.makedirs(path_hp, exist_ok=True)
        file = os.path.join(*[path_hp, "tau2.txt"])
        save2txt(self.tau2_best, file)

    def calc_posterior(self,
                       x_train: np.ndarray,
                       y_train: np.ndarray,
                       tau2: float):
        """Calculate posterior for 'tau2'."""
        def posterior(Z, sigma, y, tau2):
            # calculate 1/tau2 * I
            V0inv = np.zeros((Z.shape[1], Z.shape[1]))
            np.fill_diagonal(V0inv, 1/tau2)

            # invert likelihood matrix
            sigmainv = np.zeros(sigma.shape)
            np.fill_diagonal(sigmainv, 1/sigma.diagonal())

            # calculate posterior mean and variance
            vN = np.linalg.inv(V0inv + Z.T.dot(sigmainv).dot(Z))
            # wN = vN.dot(V0inv.dot(w0) + Z.T.dot(sigmainv).dot(y))
            wN = vN.dot(Z.T).dot(sigmainv).dot(y)
            return wN, vN

        # predict data matrix and get shape
        Z_train = self.predict_psi(x_train)
        n, p = Z_train.shape

        # get predictive mean and variance for test set
        mean, log_var = self.predict(x_train).T
        var = np.exp(log_var)

        # rescale data matrix to be zero mean
        self.Z_train_mean = Z_train.mean(axis=0).reshape(1, -1)
        Z_train = Z_train - self.Z_train_mean

        # calculate Wn and vN for posterior
        sigma = np.diag(var)
        self.wN, self.vN = posterior(Z_train, sigma, y_train, tau2)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            tau2: Optional[List[float]] = TAU2_GRID,
            normalize_y: Optional[bool] = True,
            val_ratio: Optional[float] = 0.2,
            path: Optional[str] = None,
            *args, **kwargs):
        """Fit NLM to data.

        Finds best prior variance in 'tau2' on a validation set with ratio
        'val_ratio'. With best tau2, model is fitted to full data.


        Parameters
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        tau2 : Optional[List[float]]
            Prior variances to perform grid search on.
        normalize_y : Optional[bool] = true
            Whether to standardize targets for fit.
        val_ratio : Optional[float] = 0.2
            Ratio of validation set for finding best prior variance.
        path : Optional[str] = None
            If 'path' is given, model will be loaded from 'path'.

        """
        # store some parameters
        self._model_paras["tau2"] = tau2
        self._model_paras["val_ratio"] = val_ratio

        # save unscaled y data
        y_train0 = y_train.copy()

        # standardize y data for fitting
        if normalize_y is True:
            self.y_train_std = y_train.std()
            self.y_train_mean = y_train.mean()
            y_train = (y_train - self.y_train_mean) / self.y_train_std
        else:
            self.y_train_std = 1
            self.y_train_mean = 0


        # load model from path
        if path is not None:
            self.load_weights(os.path.join(*[path, "variables", "variables"])).expect_partial()
            self.tau2_best = txt2float(os.path.join(*[path, "hyper", "tau2.txt"]))

        # calculate weights and best tau2
        else:
            if isinstance(tau2, (int, float)):
                self.tau2_best = tau2

            # find best tau2 on validation set (trained on reduced training set)
            else:
                # split train data to train/val sets
                id_train, id_val = generate_random_split(y_train, ratio=val_ratio)

                # save weights
                initial_weights = self.get_weights()

                # fit NN of subset
                super().fit(x_train[id_train, :], y_train[id_train], *args, **kwargs)

                # calculate log score for all values in tau2
                log_scores = []
                for tau2_ in tau2:
                    self.calc_posterior(x_train[id_train, :], y_train[id_train], tau2_)
                    pred = self.make_prediction(x_train[id_val, :])
                    log_scores.append(pred.logpdf(y_train0[id_val]).mean())

                # set best tau2 value -> maximizes log score
                self.tau2_best = tau2[np.argmax(log_scores)]

                # restore weights and compiler
                self.set_weights(initial_weights)
                self.recompile()

        # store best tau2
        self._model_paras["tau2_best"] = self.tau2_best

        # fit on full train data on and calc posterior
        ret_fit = super().fit(x_train, y_train, *args, **kwargs)
        self.calc_posterior(x_train, y_train, self.tau2_best)
        return ret_fit

    def pred(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make prediction for 'x'.

        Parameters
        ----------
        x : np.ndarray
            Features to predict at.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictive mean, predictive variance)

        """
        def posterior_predicitve(wN, vN, x_var, x_Z):
            x_Z = x_Z.T
            mean = wN.reshape(1,-1).dot(x_Z).ravel()
            var = x_var + x_Z.T.dot(vN).dot(x_Z)
            return mean, var.diagonal()

        # predict data matrix and substract mean
        Z = self.predict_psi(x) - self.Z_train_mean

        # get predictive variance of network
        _, log_var = self.predict(x).T
        var = np.exp(log_var)

        # calculate mean and variance (posterior predictive)
        mean, var = posterior_predicitve(self.wN, self.vN, var, Z)

        # scale predictive mean and variance by y_train
        mean = mean*self.y_train_std + self.y_train_mean
        var = var*self.y_train_std**2
        return mean, var

    def make_prediction(self, x: np.ndarray):
        """Get prediction object for x."""
        return NLMPred(self, x)


class NLMPred(PredConditionalGaussian):
    def __init__(self, model: NLM, x: np.ndarray):
        super(NLMPred, self).__init__(x.shape[0])
        assert model.pred_var is True

        # get means and variances of posterior predictive
        self.mean, self.y_var = model.pred(x)

        # make Gaussian dists
        self._make_gaussians()

    @property
    def var_aleatoric(self):
        return None

    @property
    def var_epistemic(self):
        return None

    @property
    def var_total(self):
        return self.y_var

    @property
    def pred_mean(self):
        return self.mean
