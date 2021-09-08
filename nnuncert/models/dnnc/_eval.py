import abc

import numpy as np
import pandas as pd
import scipy as sp
import scipy.integrate as spint
import scipy.interpolate as spip
import scipy.stats as spstats


from nnuncert.models.dnnc._func_deriv import clpy, compS, compS2


class PredictionBase(abc.ABC):
    def __init__(self, B, dist, betas=None, taus=None, tau2s=None, lambdas=None, dnnc_type="ridge"):
        self.B = B
        self.dist = dist
        self.dnnc_type = dnnc_type

        self.betas = betas
        self.lambdas = lambdas
        self.taus = taus
        if tau2s is not None:
            tau2s = tau2s.ravel()
        self.tau2s = tau2s

        if self.dnnc_type == "ridge":
            self.mu, self.sigma = calculate_mu_sigma(B, tau2s, betas)
        elif self.dnnc_type == "horseshoe":
            self.mu, self.sigma = calculate_mu_sigma_hs(B, betas, lambdas)
        self._lpy = None

    @abc.abstractproperty
    def lpy(self):
        pass

    @abc.abstractproperty
    def Ey(self):
        pass

    @abc.abstractproperty
    def Vary(self):
        pass

    @property
    def py(self):
        py = np.exp(self.lpy)
        np.nan_to_num(py, copy=False)
        return py


class DNNCEvaluate(PredictionBase):
    """Class to get mean and variances for psi basis functions and calc metrics."""
    def __init__(self, Bpred, dist, betas=None, taus=None, tau2s=None,
                 lambdas=None, dnnc_type="ridge"):
        super().__init__(Bpred, dist, betas=betas, taus=taus, tau2s=tau2s,
                         lambdas=lambdas, dnnc_type=dnnc_type)

        self._samples = None
        self._samples2 = None

    def score_at(self, ypred):
        self.ypred = ypred.reshape(-1, 1)
        self.pdfy = self.dist.pdf(self.ypred)
        self.zpred = self.dist.get_z(self.ypred)
        tmp = self.lpy

    @property
    def lpy(self):
        if self.ypred is None:
            raise Exception("Method 'scort_at' must be called first.")
        if self._lpy is None:
            if self.mu.ndim == 1 or self.mu.shape[1] == 1:
                self._lpy = clpy(self.zpred, self.mu, self.sigma, self.pdfy)
            else:
                lpy = []
                for j in range(self.mu.shape[1]):
                    mu_ = self.mu[:, j]
                    sigma_ = self.sigma[:, j]
                    lpy.append(clpy(self.zpred, mu_, sigma_, self.pdfy))
                self._lpy = np.array(lpy)

        return self._lpy

    @property
    def log_score(self):
        return self.lpy.diagonal().mean(axis=0)

    @property
    def rmse(self):
        y = self.ypred.ravel()
        yhat = self.Ey.values
        return np.mean((yhat - y)**2)**0.5

    @property
    def samples(self):
        if self._samples is None:
            self._samples, self._samples2 = collect_samples(self.dist,
                                                            self.mu,
                                                            self.sigma,
                                                            return_as="df")
        return self._samples

    @property
    def samples2(self):
        if self._samples2 is None:
            self._samples, self._samples2 = collect_samples(self.dist,
                                                            self.mu,
                                                            self.sigma,
                                                            return_as="df")
        return self._samples2

    @property
    def Ey(self):
        return self.samples.mean(axis=1)

    @property
    def Vary(self):
        return self.samples2.mean(axis=1) - self.Ey**2

    def get_quantiles(self, quants=[0.025, 0.05, 0.95, 0.975]):
        if self.samples.shape[1] == 1:
            return None
        else:
            return self.samples.quantile(quants, axis=1).T


class DNNCDensity(PredictionBase):
    """Class to get predictive distibutions for psi basis functions."""
    def __init__(self, B, dist, betas=None, taus=None, tau2s=None,
                 lambdas=None, dnnc_type="ridge"):
        super().__init__(B, dist, betas=betas, taus=taus, tau2s=tau2s,
                         lambdas=lambdas, dnnc_type=dnnc_type)
        self.yseq = dist.x_linspace
        self.pdfyseq = dist.pdf(self.yseq)
        self.zseq = dist.get_z(self.yseq)
        self._dists = None
        self._Ey = None
        self._Ey2 = None

    @property
    def lpy(self):
        if self._lpy is None:
            s_ = self.sigma.reshape(-1, 1)
            mu_ = self.mu.reshape(-1, 1)
            self._lpy = clpy(self.zseq, mu_, s_, self.pdfyseq)
        return self._lpy

    @property
    def Ey(self):
        if self._Ey is None:
             self._Ey = spint.trapz(self.yseq*self.py, self.yseq)
        return self._Ey

    @property
    def Ey2(self):
        if self._Ey2 is None:
             self._Ey2 = spint.trapz(self.yseq**2*self.py, self.yseq)
        return self._Ey2

    @property
    def Vary(self):
        return self.Ey2 - self.Ey**2

    def get_py_dists(self):
        if self._dists is None:
            self._dists = [Dist._from_dens(self.yseq, self.py[i])
                           for i in range(self.lpy.shape[0])]
        return self._dists


def calculate_mu_sigma(B, tau2s, betas):
    n, p = B.shape
    W = np.array([B[kk, :].dot(B[kk, :]) for kk in range(0, n)]).reshape(-1, 1)
    sigma = np.array([compS(tau2s, W[i])[1] for i in range(n)]).reshape(n, -1)
    if tau2s.size == 1:
        sigma = sigma.ravel()
    mu = sigma*(B.dot(betas.T))
    return mu, sigma


def calculate_mu_sigma_hs(B, betas, lambdas):
    BoB = B**2
    sigma = compS2(lambdas, BoB)[2]
    if betas.ndim == 1:
        betas = betas.reshape(1, -1)
    mu = sigma*(B.dot(betas.T))
    return mu, sigma


def collect_samples(dist, mu, sigma, return_as="array"):
    # overload for multiple mcmc samples
    if mu.ndim > 1 and mu.shape[1] > 1:
        samples = []
        samples2 = []
        for j in range(mu.shape[1]):
            mu_ = mu[:, j]
            sigma_ = sigma[:, j]
            s, s2 = collect_samples(dist, mu_, sigma_, return_as="array")
            samples.append(s)
            samples2.append(s2)
        samples = np.vstack(samples).T
        samples2 = np.vstack(samples2).T
        if return_as == "df":
            samples = pd.DataFrame(samples)
            samples2 = pd.DataFrame(samples2)
        return samples, samples2

    # setup for interpolation
    ip = spip.interp1d(dist.Fx, dist.x)  # reverse_interpolation

    val = 100  # set linspace for interpolation (mass in [0, 1] for cdf...)
    z = np.linspace(-val, val, 10000)
    f1 = ip(spstats.norm.cdf(z)).reshape(-1, 1)
    f12 = ip(spstats.norm.cdf(z)).reshape(-1, 1)**2
    f2 = sp.stats.norm.pdf(z.reshape(-1, 1), loc=mu.reshape(1, -1), scale=sigma.reshape(1, -1))
    f3 = f1*f2
    f4 = f12*f2

    # integrate via scipy to collect samples
    samples = spint.simps(f3, z.reshape(-1, 1), axis=0)
    samples2 = spint.simps(f4, z.reshape(-1, 1), axis=0)

    if return_as == "df":
        samples = pd.DataFrame(samples)
        samples2 = pd.DataFrame(samples2)
    return samples, samples2
