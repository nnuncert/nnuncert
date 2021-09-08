from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import os
import json
import copy

import numpy as np
import scipy.stats as spstats

from nnuncert.models._pred_base import BasePred, PredConditionalGaussian
from nnuncert.models.nlm import NLM
from nnuncert.models.pnn import PNN


class Ensemble():
    def __init__(self, N: int):
        self.num_models = N
        self._model_paras = {"num_models" : N}

    def compile(self, *args, **kwargs):
        """Compile all ensemble members."""
        # compile all ensemble members
        [m.compile(*args, **kwargs) for m in self.models]

    def save(self, path: str, *args, **kwargs):
        """Save model to path."""
        # save all ensemble members individually
        for i, m in enumerate(self.models):
            m.save(os.path.join(path, str(i)), *args, **kwargs)

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

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            path: Optional[str] = None,
            *args, **kwargs):
        """Fit all ensebmle members to data.

        Parameters
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        path : Optional[str]
            If 'path' is given, ensemble members will be loaded from 'path'.

        """
        # load/fit all ensemble members
        for i, m in enumerate(self.models):
            # 'path' given -> load ensemble member
            if path is not None:
                pathi = os.path.join(path, str(i))

            # 'path' is None -> fit ensemble member
            else:
                pathi = None

            m.fit(x_train, y_train, path=pathi, *args, **kwargs)

    def pred(self, x: np.ndarray):
        """Get predictive means and variance for all ensemble members."""
        # get predictions for all ensemble members for x
        pred = [m.make_prediction(x) for m in self.models]

        # extract means and variances
        means = np.vstack(([p.pred_mean for p in pred])).T
        vars = np.vstack(([p.var_total for p in pred])).T
        return means, vars

    def make_prediction(self, x:np.ndarray, method="gauss", *args, **kw):
        """Get prediction object for features 'x'."""
        if method == "gauss":
            return EnsPredGauss(self, x, *args, **kw)
        elif method == "gmm":
            return PredEnsGMM(self, x, *args, **kw)
        raise ValueError()


class PNNEnsemble(Ensemble):
    # disagreement! footnote 9 in Ens paper:
    # -> UPDATE: what? they calculate 'disagreement via KL divergence'
    def __init__(self, net, N: int = 5, *args, **kwargs):
        super(PNNEnsemble, self).__init__(N)
        def make_dnn(net):
            return PNN(net, *args, **kwargs)

        # make 'N' PNNs
        self.models = [make_dnn(net.clone_with_prefix(str(i)))
                       for i in range(N)]


class NLMEnsemble(Ensemble):
    def __init__(self, net, N: int = 5, *args, **kwargs):
        super(NLMEnsemble, self).__init__(N)
        def make_nlm(net):
            return NLM(net, *args, **kwargs)

        # make 'N' NLMs
        self.models = [make_nlm(net.clone_with_prefix(str(i)))
                       for i in range(N)]

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            path: Optional[str] = None,
            *args, **kwargs):
        """Fit all ensebmle members to data.

        Parameters
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        path : Optional[str]
            If 'path' is given, ensemble members will be loaded from 'path'.

        """
        # load/fit all ensemble members
        for i, m in enumerate(self.models):
            # 'path' given -> load ensemble member
            if path is not None:
                pathi = os.path.join(path, str(i))

            # 'path' is None -> fit ensemble member
            else:
                pathi = None

            m.fit(x_train, y_train, path=pathi, *args, **kwargs)
        self._model_paras["tau2"] = copy.copy(self.models[0]._model_paras["tau2"])
        self._model_paras["val_ratio"] = copy.copy(self.models[0]._model_paras["val_ratio"])


class EnsPredGauss(PredConditionalGaussian):
    def __init__(self, ens, x):
        self.xlen = x.shape[0]

        # make Gaussians from means and variances
        self.means, self.vars = ens.pred(x)
        self._make_gaussians()

    @property
    def var_epistemic(self):
        return None

    @property
    def var_aleatoric(self):
        return None

    @property
    def var_total(self):
        return (self.vars + self.means**2).mean(axis=1) - (self.pred_mean**2)

    @property
    def pred_mean(self):
        return self.means.mean(axis=1).ravel()


# ███    ███  █████  ██    ██ ██████  ███████ 
# ████  ████ ██   ██  ██  ██  ██   ██ ██      
# ██ ████ ██ ███████   ████   ██████  █████ 
# ██  ██  ██ ██   ██    ██    ██   ██ ██    
# ██      ██ ██   ██    ██    ██████  ███████ 
class GMM():
    def __init__(self, means, vars, weights=None):
        assert len(means) == len(vars) <= 10
        self.means = np.array(means)
        self.vars = np.array(vars)
        if weights is None:
            self.weights = np.array([1/self.n]*self.n)
        else:
            self.weights = weights

    @property
    def components(self):
        return list(zip(self.weights, self.means, self.vars**0.5))

    @property
    def bounds(self):
        left = min([spstats.norm.ppf(0.001, m, v) for (_, m, v) in self.components])
        right = max([spstats.norm.ppf(0.999, m, v) for (_, m, v) in self.components])
        return (left, right)

    @property
    def E(self):
        return np.mean(self.means)

    @property
    def Var(self):
        return np.mean(self.means**2 + self.vars) - self.E**2

    @property
    def n(self):
        return len(self.means)

    def pdf(self, x):
        return np.sum([w*spstats.norm.pdf(x, m, s) for (w, m, s) in self.components], axis=0)

    def cdf(self, x):
        return np.sum([w*spstats.norm.cdf(x, m, s) for (w, m, s) in self.components], axis=0)

    def ppf(self, q):
        assert q <= 0.05 or q >= 0.95
        if q < 0.5:
            left = min([spstats.norm.ppf(0.001, m, v) for (_, m, v) in self.components])
            x0 = np.linspace(left, min(self.means), 100)
        else:
            right = max([spstats.norm.ppf(0.999, m, v) for (_, m, v) in self.components])
            x0 = np.linspace(max(self.means), right, 100)

        Fx0 = self.cdf(x0)
        return spinterpol.interp1d(Fx0, x0)(q)


class PredEnsGMM(BasePred):
    def __init__(self, ens, x):
        self.xlen = x.shape[0]
        self.means, self.vars = ens.pred(x)
        self.gmms = [GMM(m, v) for (m, v) in list(zip(self.means, self.vars))]

    @property
    def var_total(self):
        return (self.vars + self.means**2).mean(axis=1) - (self.pred_mean**2)

    @property
    def pred_mean(self):
        return self.means.mean(axis=1)

    def pdf(self, y):
        return np.array([gmm.pdf(y_) for (gmm, y_) in list(zip(self.gmms, y))])

    def logpdf(self, y):
        return np.log(self.pdf(y))

    def cdf(self, y):
        return np.array([gmm.cdf(y_) for (gmm, y_) in list(zip(self.gmms, y))])

    def ppf(self, q, *args, **kwds):
        return np.array([gmm.ppf(q) for gmm in self.gmms])

    def pdfi(self, i, y):
        return self.gmms[i].pdf(y)

    def cdfi(self, i, y):
        return self.gmms[i].cdf(y)
