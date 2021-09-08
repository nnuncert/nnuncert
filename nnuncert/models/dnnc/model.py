from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import os
import warnings

import numpy as np

from nnuncert.models._network import MakeNet
from nnuncert.models._model_base import BaseModel
from nnuncert.models._pred_base import BasePredKDE
from nnuncert.models.dnnc._dnnc import DNNCRidgeIWLS, DNNCHorseshoeIWLS
from nnuncert.models.dnnc._eval import DNNCDensity, DNNCEvaluate
from nnuncert.utils.dist import Dist
from nnuncert.utils.io import load_obj, save_obj


class DNNCModel(BaseModel):
    def __init__(self,
                 net: MakeNet,
                 dnnc_type: Optional[str] = "ridge",
                 *args, **kwargs):
        assert net.pred_var is False, "DNNC requires NN fit with MSE."
        assert dnnc_type in ["ridge", "horseshoe"], \
            "'dnnc_type' must be ridge or horseshoe"

        super(DNNCModel, self).__init__(net, *args, **kwargs)
        self.dnnc_type = dnnc_type
        self._model_paras["dnnc_type"] = dnnc_type


    def save(self, path: str, *args, **kwargs):
        """Save model to path."""
        # save NN weights
        super().save(path, *args, **kwargs)

        # remove memory expensive attributes
        self.dnnc._set_means_only()  # remove samples for parameters betas, ...
        self.B = None  # can be retrieved easily with training x

        # save dnnc data
        save_obj(self.dnnc, os.path.join(path, "dnnc_" + self.dnnc_type))

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            dist: Dist,
            verbose: Optional[int] = 0,
            fit_z_train: Optional[bool] = True,
            set_dnnc: Optional[bool] = True,
            path: Optional[str] = None,
            dnnc_kwargs: Optional[dict] = {},
            *args, **kwargs):
        """Fit DNNC to data.

        Parameters
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        dist : Dist
            Estimated KDE of response.
        verbose : Optional[int]
        fit_z_train : Optional[bool] = True
            Whether to fit NN on targets transformed by F_Y, True by default. If
            False, NN will be fit to standardized targets.
        set_dnnc : Optional[bool] = True
            Whether to automatically run DNNC after the NN fit.
        path : Optional[str]
            If given, NN and DNNC data will be loaded from 'path', no fitting.
        dnnc_kwargs : Optional[dict]
            Parameters to pass to DNNC sampling.

        """
        self._model_paras["fit_z_train"] = fit_z_train
        dnnc_kwargs["dist"] = dist
        dnnc_kwargs["verbose"] = verbose
        kwargs["verbose"] = verbose

        # load weights
        if path is not None:
            p = os.path.join(*[path, "variables", "variables"])
            self.load_weights(p).expect_partial()

        # fit NN to data
        else:
            # fit data on transformed z = F_y^(-1)(y)
            if fit_z_train is True:
                self.fitted_on = "z = transformed y"

                # get transformed targets and fit NN to it
                z_train = dist.get_z(y_train)
                super().fit(x_train, z_train, *args, **kwargs)

            # fit data on standardized y = (y_train - y_train_mean) / y_train_std
            else:
                self.fitted_on = "standardized y"
                y_train_std = y_train.std()
                y_train_mean = y_train.mean()
                super().fit(x_train, (y_train - y_train_mean) / y_train_std,
                            *args, **kwargs)

        # predict data matrix and get indices to be cleaned
        self.psi_clean, self.psi_idx = self.predict_psi(x_train)

        # if path is given, load dnnc
        if path is not None:
            self.dnnc = load_obj(os.path.join(path, "dnnc_" + self.dnnc_type))
            self.dnnc.B = self.psi_clean

        # compute dnnc
        else:
            if set_dnnc is True:
                if self.dnnc_type == "ridge":
                    self.dnnc = DNNCRidgeIWLS(self.psi_clean, y_train, **dnnc_kwargs)
                else:
                    self.dnnc = DNNCHorseshoeIWLS(self.psi_clean, y_train,
                                                **dnnc_kwargs)

        # keep track of model parameters
        self._model_paras["J"] = self.dnnc.J

        if self.dnnc_type == "ridge":
            self._model_paras["theta"] = self.dnnc.theta
            self._model_paras["tau2start"] = self.dnnc.tau2start
        else:
            self._model_paras["taustart"] = self.dnnc.taustart

    def predict_psi_test(self, x: np.ndarray):
        """Predict psi values for some test data.

        Psi values will be cleaned the same way as training data.
        """
        # predict hidden output layer values
        psi = super().predict_psi(x, clean=False)[0]

        # use only columns identified in training process
        psi = psi[:, self.psi_idx]
        return psi

    def make_prediction(self, x: np.ndarray, *args, **kwargs):
        """Get prediction object for x."""
        return DNNCPred(self, x, *args, **kwargs)


class DNNCRidge(DNNCModel):
    def __init__(self, net: MakeNet, *args, **kwargs):
        super(DNNCRidge, self).__init__(net, dnnc_type="ridge", *args, **kwargs)


class DNNCHorseshoe(DNNCModel):
    def __init__(self, net: MakeNet, *args, **kwargs):
        warnings.warn("DNNC horseshoe may not function properly")
        super(DNNCHorseshoe, self).__init__(net, dnnc_type="horseshoe", *args, **kwargs)


class DNNCPred(BasePredKDE):
    def __init__(self,
                 model: DNNCModel,
                 x: np.ndarray,
                 psi_test: Optional[np.ndarray] = None):
        super(DNNCPred, self).__init__(x.shape[0])

        # predict data matrix for test data
        psi = psi_test
        if psi_test is None:
            psi = model.predict_psi_test(x)

        # compute densities, expected value, variance
        dnnc = model.dnnc
        if model.dnnc_type == "ridge":
            self.eval = DNNCEvaluate(psi, dnnc.dist, betas=dnnc.betahat,
                                     tau2s=dnnc.tau2hat,
                                     dnnc_type=model.dnnc_type)
            self.dens = DNNCDensity(psi, dnnc.dist, betas=dnnc.betahat,
                                    tau2s=dnnc.tau2hat,
                                    dnnc_type=model.dnnc_type)
        elif model.dnnc_type == "horseshoe":
            self.eval = DNNCEvaluate(psi, dnnc.dist, betas=dnnc.betahat,
                                     taus=dnnc.tauhat, lambdas=dnnc.lambdahat,
                                     dnnc_type=model.dnnc_type)
            self.dens = DNNCDensity(psi, dnnc.dist, betas=dnnc.betahat,
                                    taus=dnnc.tauhat, lambdas=dnnc.lambdahat,
                                    dnnc_type=model.dnnc_type)
        self.mean = np.array(self.eval.Ey)
        self.y_var = np.array(self.eval.Vary)
        py = np.exp(self.dens.lpy)
        np.nan_to_num(py, copy=False)

        # create proper prob. dist. for densities
        self.dists = [Dist._from_fx(dnnc.dist.x_linspace, p) for p in py]

    def log_score(self, y):
        """Compute log_score."""
        assert len(y) == self.xlen, "x, y length missmatch."

        # evaluate
        self.eval.score_at(y)

        # get log scores from eval
        self._log_scores = self.eval.lpy.diagonal()

        return self._log_scores.mean()

    def log_score_x(self, y, frac=0.99):
        N = int(len(y)*frac)
        return pd.Series(self.logpdf(y)).nlargest(N).mean()

    @property
    def var_aleatoric(self):
        """Split of variance not available."""
        return None

    @property
    def var_epistemic(self):
        """Split of variance not available."""
        return None

    @property
    def var_total(self):
        """Get total predictive variance."""
        return self.y_var

    @property
    def pred_mean(self):
        """Get predictive mean."""
        return self.mean
