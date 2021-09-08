from typing import Tuple, Union, Iterable, List, Callable
import warnings

import numpy as np
import scipy.integrate as spint
import scipy.interpolate as spip
import scipy.stats as spstats

from nnuncert.utils.dist.ssv import ssvkernel


def kde_ssv(y: Union[list, np.ndarray], x: Union[None, np.ndarray] = None,
            set_x: bool = True, ma: Union[None, int, float] = None,
            mb: Union[None, int, float] = None,
            bound_left: Union[None, int, float] = None,
            bound_right: Union[None, int, float] = None, points: int = 500,
            **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    # make response to 1d array
    y = y.ravel()

    # estimate search array (where to get kde values) if not given
    if x is None and set_x is True:
        xmin, xmax = min(y), max(y)

        # set margins left (ma) and right (mb) to std if not given
        sd = np.std(y)
        margins = [sd, sd]
        if ma is not None:
            margins = [ma, mb]

        # set problem specific bounds, overwrites margins
        if bound_left is not None:
            left = bound_left
        else:
            left = xmin - margins[0]
        if bound_right is not None:
            right = bound_right
        else:
            right = xmax + margins[1]

        # make linspace (where to get kde values)
        x = np.linspace(left, right, points)

    # calculate actual kde
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # disable some warnings in ssvkernel
        fx, x = ssvkernel(y, x, **kwargs)[:2]

    return x, fx

def get_cdf(x: np.ndarray, fx: np.ndarray, tol: float = 0.001
            ) -> Tuple[np.ndarray, np.ndarray, float]:
    # create empty cdf array and set first val to zero
    Fx = np.zeros(fx.shape)
    Fx[0] = 0

    # integrate pdf values
    for i in range(len(fx)):
        Fx[i] = np.trapz(fx[:i+1], x[:i+1])

    # check that area under fx vals is close to 1
    eps = 1 - Fx[-1]
    # assert eps < tol, "tolerance breached" + str(1 - Fx[-1])

    # values greater than one not allowed
    Fx[Fx > 1] = 1

    # set last value in cdf to 1
    Fx[-1] = 1

    # make unique cdf and support values
    Fx, Fx_unique = np.unique(Fx, return_index=True)
    x = x[Fx_unique]
    x, x_unique = np.unique(x, return_index=True)
    Fx = Fx[x_unique]

    # ensure monotonically increasing
    if np.sum(np.diff(Fx) < 0) > 0:
        x_, Fx_ = [x[0]], [Fx[0]]
        for i in range(1, len(Fx)):
            if Fx[i] > Fx_[-1]:
                Fx_.append(Fx[i])
                x_.append(x[i])
        return x_, Fx_, eps

    return x, Fx, eps


class Dist():
    def __init__(self, x: np.ndarray, fx: np.ndarray):
        # store init values
        self.x0 = x
        self.fx0 = fx

        # make pdf interpolator
        self._ip_fx = spip.interp1d(x, fx, bounds_error=False, fill_value=0)  # was Cubic Spline

        # get cdf from pdf via integration and clean
        self.x, self.Fx, self.eps = get_cdf(x, fx)

        # cast to np array
        self.x, self.Fx = np.array(self.x), np.array(self.Fx)

        # make cdf and ppf interpolators
        self._ip_Fx = spip.interp1d(self.x, self.Fx, bounds_error=False, fill_value=(0, 1))
        self._ip_ppf = spip.interp1d(self.Fx, self.x, bounds_error=False, fill_value=(0, 1))

        # empty internal moments (filled by properties)
        self._E = None
        self._E2 = None

    def ppf(self, Fx):
        """Get ppf value for 'Fx'."""
        # Fx = self.cdf(self.x)
        # ip = spip.PchipInterpolator(Fx, self.x)
        return self._ip_ppf(Fx)

    def cdf(self, x):
        """Get cdf value for 'x'."""
        eps = 1e-15  # hacked
        val = self._ip_Fx(x)
        val[val < 0] = 0
        val[val > 1- eps] = 1
        return val

    def pdf(self, x):
        """Get cdf value for 'x'."""
        val = self._ip_fx(x)
        val[val < 0] = 0
        return val

    def get_z(self, y):
        """Get z = spstats.norm.ppf(self.cdf(y))."""
        Fx = self.cdf(y)
        return spstats.norm.ppf(Fx)

    def get_support(self, eps=1e-5):
        supp_left = self.x[np.argmin(self.Fx <= eps) - 1]
        supp_right = self.x[np.argmax(self.Fx >= 1 - eps)]
        return (supp_left, supp_right)

    def save(self, path, *arg, **kwargs):
        """Save dist initializers to path."""
        np.save(path, self.get_core_vals(), *args, **kwargs)

    def get_core_vals(self):
        """Get values used to initialize dist."""
        return np.vstack((self.x0, self.fx0)).T

    @property
    def x_linspace(self):
        return np.linspace(min(self.x), max(self.x), 200)

    @property
    def E(self):
        """Get first moment via integration."""
        if self._E is None:
            fx = self.pdf(self.x)
            self._E = spint.trapz(self.x*fx, self.x)
        return self._E

    @property
    def E2(self):
        """Get second moment via integration."""
        if self._E2 is None:
            fx = self.pdf(self.x)
            self._E2 = spint.trapz(self.x**2*fx, self.x)
        return self._E2

    @property
    def Var(self):
        """Get variance."""
        return self.E2 - self.E**2

    @classmethod
    def _from_values(cls, vals: np.ndarray,
                     method: Union["ssv", "gauss", "uniform"] = "ssv",
                     noise: Union[int, float] = 0,
                     linspace: Union[None, np.ndarray] = None, **kwargs):
        vals = vals.ravel()
        # add noise to response for smooth kde
        if noise != 0:
            vals = vals + np.random.normal(0, noise, len(vals))

        # estimate kde via ssv kernel
        if method == "ssv":
            x, fx = kde_ssv(vals, **kwargs)
            return cls._from_fx(x, fx)

        # estimate gaussian kde
        elif method == "gauss":
            kde = spstats.gaussian_kde(vals, **kwargs)
            sd = np.std(vals)
            if linspace is None:
                linspace = np.linspace(min(vals) - sd, max(vals) + sd, 1000)
            fx = kde(linspace)
            return cls._from_fx(linspace, fx)

        # estimate uniform kde
        elif method == "uniform":
            linspace = np.linspace(min(y), max(y), 100)
            fx = np.ones(100) * 1 / (max(y) - min(y))
            return cls._from_fx(linspace, fx)

    @classmethod
    def _from_fx(cls, x, fx):
        return cls(x, fx=fx)

    @classmethod
    def _from_file(cls, file):
        x, fx = np.load(file).T
        return cls._from_fx(x, fx)
