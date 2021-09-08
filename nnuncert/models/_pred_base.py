from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as spstats
import scipy.integrate as spint
import properscoring as ps


class MCPredictions(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    def quantile(self, q):
        return np.quantile(self, q, axis=1)

    def contains(self, y):
        return (y >= self.min(axis=1)) & (y <= self.max(axis=1))

    def in_95(self, y):
        q2p5, q97p5 = np.quantile(self, [0.025, 0.975], axis=1)
        ratio = sum((y >= q2p5) & (y <= q97p5)) / len(y)
        return ratio, (q97p5-q2p5).mean()

    def median(self, *args, **kwargs):
        return np.median(self, *args, **kwargs)

    def map(self, pts_grid=100):
        def find_peak(x, pts_grid=pts_grid):
            x = np.array(x).ravel()
            kde = sp.stats.gaussian_kde(x)
            x_grid = np.linspace(min(x), max(x), pts_grid)
            map = np.argmax(kde(x_grid))
            return x_grid[map]
        return np.array([find_peak(y_) for y_ in self.lst])

    @property
    def lst(self):
        return list(self)

    @property
    def np(self):
        return np.array(self)


class BasePred():
    def __init__(self, xlen):
        self.xlen = xlen

    def pdf(self, y):
        raise NotImplementedError()

    def logpdf(self, y):
        raise NotImplementedError()

    def cdf(self, y):
        raise NotImplementedError()

    def ppf(self, q: float):
        raise NotImplementedError("Need to fix bug in Dist -> Fx first...")

    def pdfi(self, i: int, x):
        raise NotImplementedError()

    def cdfi(self, i: int, x):
        raise NotImplementedError()

    @property
    def pred_mean(self):
        raise NotImplementedError()

    @property
    def var_aleatoric(self):
        """Get pred. aleatoric variance."""
        return np.zeros(self.xlen)

    @property
    def var_epistemic(self):
        return np.zeros(self.xlen)

    @property
    def var_total(self):
        return self.var_aleatoric + self.var_epistemic

    @property
    def std_total(self):
        return self.var_total**0.5

    @property
    def q2p5(self):
        return self.ppf(0.025)

    @property
    def q97p5(self):
        return self.ppf(0.975)

    def marginals(self, y0, recalc: bool = True):
        if recalc is False and hasattr(self, "_marginals"):
            return self._marginals
        self._marginals = np.array([self.pdf(np.ones(self.xlen) * y_) for y_ in y0]).mean(axis=1)
        return self._marginals

    # ================================= SCORES =================================
    def rmse(self, y):
        return np.mean((self.pred_mean - y)**2)**0.5

    def log_score(self, y):
        assert len(y) == self.xlen, "x, y length missmatch."
        self._log_scores = self.logpdf(y)
        return self._log_scores.mean()

    def log_score_x(self, y, frac: float = 0.99):
        N = int(len(y)*frac)
        return pd.Series(self.logpdf(y)).nlargest(N).mean()

    def crps(self, y):
        raise NotImplementedError()

    def picp(self, y):
        return np.mean((y >= self.q2p5) & (y <= self.q97p5))

    @property
    def mpiw(self):
        return np.mean(self.q97p5 - self.q2p5)


class PredConditionalGaussian(BasePred):
    def pdf(self, y):
        return self.gaussians.pdf(y)

    def pdfi(self, i: int, y):
        mean, std = self.pred_mean[i], self.std_total[i]
        return spstats.norm.pdf(y, mean, std)

    def logpdf(self, y):
        return self.gaussians.logpdf(y)

    def cdf(self, y):
        return self.gaussians.cdf(y)

    def cdfi(self, i: int, y):
        mean, std = self.pred_mean[i], self.std_total[i]
        return spstats.norm.cdf(y, mean, std)

    def ppf(self, q: float):
        return self.gaussians.ppf(q)

    def crps(self, y):
        self._crps = np.array([ps.crps_gaussian(x, mu=m, sig=s)
                               for (x, m, s) in list(zip(y,
                                                         self.pred_mean,
                                                         self.std_total))])
        return self._crps.mean()

    def _make_gaussians(self):
        class Gaussian():
            def __init__(self, mean, std_total):
                self.params = np.vstack((mean, std_total)).T.tolist()

            def pdf(self, y, *args, **kwds):
                return np.array([spstats.norm.pdf(y_, m, s, *args, **kwds)
                                 for y_, (m, s) in list(zip(y, self.params))])

            def logpdf(self, y, *args, **kwds):
                return np.array([spstats.norm.logpdf(y_, m, s, *args, **kwds)
                                 for y_, (m, s) in list(zip(y, self.params))])

            def cdf(self, y, *args, **kwds):
                return np.array([spstats.norm.cdf(y_, m, s, *args, **kwds)
                                 for y_, (m, s) in list(zip(y, self.params))])

            def ppf(self, q, *args, **kwds):
                return np.array([spstats.norm.ppf(q, m, s, *args, **kwds)
                                 for (m, s) in self.params])

        self.gaussians = Gaussian(self.pred_mean, self.std_total)


class BasePredKDE(BasePred):
    def pdf(self, y):
        return np.array([d.pdf(y_) for (d, y_) in list(zip(self.dists, y))])

    def pdfi(self, i, y):
        return self.dists[i].pdf(y)

    def logpdf(self, y):
        return np.log(self.pdf(y))

    def cdf(self, y):
        return np.array([d.cdf(y_) for (d, y_) in list(zip(self.dists, y))])

    def cdfi(self, i, y):
        return self.dists[i].cdf(y)

    def ppf(self, q):
        return np.array([d.ppf(q) for d in self.dists])

    def crps(self, y, N=100, eps=1e-5):
        # def get_support(self, eps=1e-5):
        # supp_left = np.argmin(self.Fx < eps) - 1
        # supp_right = np.argmax(self.Fx > 1 - eps)
        # return (supp_left, supp_right)
        assert len(y) == self.xlen, "x, y length missmatch."
        def calc_crps(dist, y):
            supp_left, supp_right = dist.get_support(eps)
            lhs = 0
            rhs = 0

            if y >= supp_left:
                ls_lhs = np.linspace(supp_left, y, N)
                val_lhs = dist.cdf(ls_lhs)**2
                lhs = spint.simps(val_lhs, ls_lhs)

            if y <= supp_right:
                ls_rhs = np.linspace(y, supp_right, N)
                val_rhs = (dist.cdf(ls_rhs) - 1)**2
                rhs = spint.simps(val_rhs, ls_rhs)

            return lhs + rhs

        self._crps = np.array([calc_crps(d, y_)
                                for (d, y_) in list(zip(self.dists, y))])
        return self._crps.mean()
