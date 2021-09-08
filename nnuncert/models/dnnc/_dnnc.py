from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import time
from abc import abstractmethod

import numpy as np
from tqdm import tqdm

from nnuncert.models.dnnc._sampling import genbeta, genbeta2, gentau, gentau2, genlambda
from nnuncert.models.dnnc._func_deriv import compdS, compdS2
from nnuncert.utils.dist import Dist


class DNNC():
    def __init__(self,
                 B: np.ndarray,
                 y: np.ndarray,
                 dist: Optional[Dist] = None,
                 dist_method: Optional[Union[str, Callable]] = "ssv",
                 name: Optional[str] = None,
                 rng: Union[None, int, np.random.Generator] = None,
                 dtype: Optional = np.float64):
        self.rng = np.random.default_rng(rng)
        self.dtype = dtype
        if name is None:
            name = "dnnc"  # + time.strftime("%y%m%d-%H:%M", time.time())

        # make dist from 'dist_method' if not given
        self.name = name
        if dist is None:
            if callable(dist_method):
                dist = dist_method(y)
            elif dist_method == "ssv":
                dist = Dist._from_values(y, method="ssv", set_x=True)
            elif dist_method == "gauss":
                dist = Dist._from_values(y, method="gauss")
            else:
                raise ValueError("dist_method not recognized.")
        self.dist = dist

        self.B = B.astype(dtype)
        self.y = y.astype(dtype).reshape(-1, 1)


class DNNCMCMC(DNNC):
    def __init__(self,
                 B: np.ndarray,
                 y: np.ndarray,
                 J: Optional[List[int]] = [1000, 3000],
                 verbose: Optional[int] = 0,
                 save_samples: Optional[bool] = True,
                 *args, **kwargs):
        self.verbose = 0
        self.save_samples = save_samples
        if verbose == 1:
            self.verbose = verbose
        super(DNNCMCMC, self).__init__(B, y, *args, **kwargs)
        self.J = J

    @property
    def disable_pbar(self):
        """Disable progressbar if self.verbose == 0."""
        if self.verbose == 0:
            return True
        return False

    def sample_mcmc(self, *args, **kwargs):
        """Sample MCMC samples."""
        if self.J[0] > 0:
            self.timestampe_start = time.time()
            self._sample_wrapper(*args, **kwargs)
            self.time_mcmc = time.time() - self.timestampe_start

    @property
    def mcmc_samples(self):
        """Get number of samples used for inference."""
        return self.J[-1]

    @property
    def mcmc_burnin(self):
        """Get number of burn_in samples."""
        return self.J[0]


class DNNCRidge():
    parameters = ["beta", "tau2"]
    dnnc_type = "ridge"

    @property
    @abstractmethod
    def betahat(self):
        pass

    @property
    @abstractmethod
    def tau2hat(self):
        pass


class DNNCHorseshoe():
    parameters = ["beta", "tau", "lambda"]
    dnnc_type = "horseshoe"

    @property
    @abstractmethod
    def betahat(self):
        pass

    @property
    @abstractmethod
    def tauhat(self):
        pass

    @property
    @abstractmethod
    def lambdahat(self):
        pass


class DNNCRidgeIWLS(DNNCMCMC, DNNCRidge):
    def __init__(self,
                 B: np.ndarray,
                 y: np.ndarray,
                 theta: Optional[float] = 2.5,
                 tau2start: Optional[float] = 10,
                 start_mcmc: Optional[bool] = True,
                 *args, **kwargs):
        super(DNNCRidgeIWLS, self).__init__(B, y, *args, **kwargs)
        self.theta = theta
        self.tau2start = tau2start
        self._means_only = False
        if start_mcmc is True:
            # initialize values for parameters
            betastart = np.zeros(B.shape[1]).astype(self.dtype)
            self.sample_mcmc(tau2start, betastart, theta)

    def resample(self):
        """Resample MCMC."""
        betastart = np.zeros(self.B.shape[1]).astype(self.dtype)
        self.sample_mcmc(self.tau2start, betastart, self.theta)

    def _sample_wrapper(self, tau2new, betanew, theta):
        # initialization
        n, p = self.B.shape
        tBB = self.B.T.dot(self.B)
        W = np.array([self.B[kk, :].dot(self.B[kk, :]) for kk in range(0, n)])
        W = W.reshape(-1, 1)
        z = self.dist.get_z(self.y)
        S2 = compdS(tau2new, W)[0]

        # burn in
        betanew, tau2new = self._sample(self.J[0], z, self.B, tBB, S2, W,
                                        tau2new, betanew, theta, burn_in=True)

        # collect mcmc samples
        betanew, tau2new = self._sample(self.J[1], z, self.B, tBB, S2, W,
                                        tau2new, betanew, theta, burn_in=False)

        if self.save_samples is False:
            self._set_means_only()

    def _set_means_only(self):
        self._means_only = True
        self._betahat = self.betas.mean(axis=0)
        self._tau2hat = self.tau2s.mean(axis=0)
        self.betas = None
        self.tau2s = None

    def _sample(self, num_samples, z, B, tBB, S2, W, tau2new, betanew, theta,
                burn_in=False):
        if burn_in is False:
            self.betas = []
            self.tau2s = []

        self.tau2accs = []

        t0 = time.time()
        for j in tqdm(range(num_samples), desc="Burn in = " + str(burn_in), position=0, leave=True, disable=self.disable_pbar):
            # sample beta
            betanew = genbeta(z, np.log(tau2new), B, tBB, S2, rng=self.rng)

            # sample tau2
            tau2new, tau2accept, S2 = gentau2(tau2new, W, B, z, betanew, theta,
                                              rng=self.rng)

            if burn_in is False:
                self.betas.append(np.array(betanew).ravel())
                self.tau2s.append(np.array(tau2new).ravel())
                self.tau2accs.append(tau2accept)

        if burn_in is False:
            self.betas = np.array(self.betas)
            self.tau2s = np.array(self.tau2s)
            self.tau2accs = np.array(self.tau2accs)

        return betanew, tau2new

    @property
    def betahat(self):
        if self._means_only is True:
            return self._betahat
        return self.betas.mean(axis=0)

    @property
    def tau2hat(self):
        if self._means_only is True:
            return self._tau2hat
        return self.tau2s.mean(axis=0)


class DNNCHorseshoeIWLS(DNNCMCMC, DNNCHorseshoe):
    def __init__(self,
                 B: np.ndarray,
                 y: np.ndarray,
                 taustart: Optional[float] = 1,
                 start_mcmc: Optional[bool] = True,
                 *args, **kwargs):
        super(DNNCHorseshoeIWLS, self).__init__(B, y, *args, **kwargs)
        self.taustart = taustart
        self._means_only = False

        if start_mcmc is True:
            # initialize values for parameters
            betastart = np.zeros(B.shape[1]).astype(self.dtype)
            lambdastart = np.ones((B.shape[1], 1)).astype(self.dtype)
            self.sample_mcmc(taustart, betastart, lambdastart)

    def resample(self):
        # initialize values for parameters
        betastart = np.zeros(self.B.shape[1]).astype(self.dtype)
        lambdastart = np.ones((self.B.shape[1], 1)).astype(self.dtype)
        self.sample_mcmc(self.taustart, betastart, lambdastart)

    def _sample_wrapper(self, taunew, betanew, lambdanew):      
        # initialization
        n, p = self.B.shape
        tBB = self.B.T.dot(self.B)
        BoB = self.B**2
        W = np.array([self.B[kk, :].dot(self.B[kk, :]) for kk in range(0, n)])
        W = W.reshape(-1, 1)
        z = self.dist.get_z(self.y)
        S2, dS2, ddS2 = compdS2(lambdanew, BoB)[1:4]

        # burn in
        betanew, taunew, lambdanew = \
            self._sample(self.J[0], z, self.B, tBB, BoB, S2, dS2, ddS2, W,
                         taunew, betanew, lambdanew, burn_in=True)
        # collect mcmc samples
        betanew, taunew, lambdanew = \
            self._sample(self.J[1], z, self.B, tBB, BoB, S2, dS2, ddS2, W,
                         taunew,betanew, lambdanew, burn_in=False)

        if self.save_samples is False:
            self._set_means_only()

    def _set_means_only(self):
        self._means_only = True
        self._betahat = self.betas.mean(axis=0)
        self._tauhat = self.taus.mean(axis=0)
        self._lambdahat = np.nanmean(self.lambdas, axis=0)
        self.betas = None
        self.taus = None
        self.lambdas = None

    def _sample(self, num_samples, z, B, tBB, BoB, S2, dS2, ddS2, W, taunew,
                betanew, lambdanew, burn_in=False):
        if burn_in is False:
            self.betas = []
            self.taus = []
            self.tauaccs = []
            self.lambdas = []
            self.lambdaaccs = []

        for j in tqdm(range(num_samples), desc="Burn in = " + str(burn_in), position=0, leave=True, disable=self.disable_pbar):
            # sample beta
            betanew = genbeta2(z, B, tBB, S2, lambdanew, rng=self.rng)

            # sample tau
            taunew, tauaccept = gentau(taunew, lambdanew, rng=self.rng)

            # samples lambdas
            lambdanew, lambdaacc, tmp, S2, dS2, ddS2 = \
                genlambda(lambdanew, S2, dS2, ddS2, B, BoB, z, betanew, taunew,
                          rng=self.rng)

            if burn_in is False:
                self.betas.append(np.array(betanew).ravel())
                self.taus.append(np.array(taunew).ravel())
                self.tauaccs.append(tauaccept)
                self.lambdas.append(lambdanew.ravel())
                self.lambdaaccs.append(lambdaacc)

        if burn_in is False:
            self.betas = np.array(self.betas)
            self.taus = np.array(self.taus)
            self.tauaccs = np.array(self.tauaccs)
            self.lambdas = np.array(self.lambdas)
            self.lambdaaccs = np.array(self.lambdaaccs)

        return betanew, taunew, lambdanew

    @property
    def betahat(self):
        if self._means_only is True:
            return self._betahat
        return self.betas.mean(axis=0)

    @property
    def tauhat(self):
        if self._means_only is True:
            return self._tauhat
        return self.taus.mean(axis=0)

    @property
    def lambdahat(self):
        if self._means_only is True:
            return self._lambdahat
        return np.nanmean(self.lambdas, axis=0)
