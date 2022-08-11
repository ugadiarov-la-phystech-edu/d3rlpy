import math
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Normal
import scipy.optimize as so
import numpy as np

def gumbel_pdf(x, loc, scale):
    """Returns the value of Gumbel's pdf with parameters loc and scale at x .
    """
    # substitute
    z = (x - loc) / scale

    return (1. / scale) * (torch.exp(-(z + (torch.exp(-z)))))


def gumbel_cdf(x, loc, scale):
    """Returns the value of Gumbel's cdf with parameters loc and scale at x.
    """
    return torch.exp(-torch.exp(-(x - loc) / scale))


def trunc_GBL(p, x):
    threshold = p[0]
    loc = p[1]
    scale = p[2]
    x1 = x[x < threshold]
    nx2 = len(x[x >= threshold])
    L1 = (-torch.log((gumbel_pdf(x1, loc, scale) / scale))).sum()
    L2 = (-torch.log(1 - gumbel_cdf(threshold, loc, scale))) * nx2
    # print x1, nx2, L1, L2
    return L1 + L2

class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def sample_n(self, n: int) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_n_with_log_prob(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        pass


class GaussianDistribution(Distribution):
    _raw_loc: torch.Tensor
    _mean: torch.Tensor
    _std: torch.Tensor
    _dist: Normal

    def __init__(
        self,
        loc: torch.Tensor,
        std: torch.Tensor,
        raw_loc: Optional[torch.Tensor] = None,
    ):
        self._mean = loc
        self._std = std
        if raw_loc is not None:
            self._raw_loc = raw_loc
        self._dist = Normal(self._mean, self._std)

    def sample(self) -> torch.Tensor:
        return self._dist.rsample().clamp(-1.0, 1.0)

    def sample_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.sample()
        return y, self.log_prob(y)

    def sample_without_squash(self) -> torch.Tensor:
        assert self._raw_loc is not None
        return Normal(self._raw_loc, self._std).rsample()

    def sample_n(self, n: int) -> torch.Tensor:
        return self._dist.rsample((n,)).clamp(-1.0, 1.0)

    def sample_n_with_log_prob(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sample_n(n)
        return x, self.log_prob(x)

    def sample_n_without_squash(self, n: int) -> torch.Tensor:
        assert self._raw_loc is not None
        return Normal(self._raw_loc, self._std).rsample((n,))

    def mean_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._mean, self.log_prob(self._mean)

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(y).sum(dim=-1, keepdims=True)

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std

class GumbelDistribution(GaussianDistribution):
        def __init__(self, loc, std, raw_loc,
                     probs=None, temperature=1):
            super().__init__(loc, std, raw_loc)

            self.probs = probs
            self.eps = 1e-20
            self.temperature = temperature

        def sample_gumbel(self):
            U = super().sample()
            U.uniform_(0, 1)
            to_gumbel =  -torch.log(-torch.log(U + self.eps) + self.eps)
       #     raise Exception(to_gumbel)
          #  U.uniform_(0, 1)
            return to_gumbel

        def gumbel_softmax_sample(self):
            """ Draw a sample from the Gumbel-Softmax distribution. The returned sample will be a probability distribution
            that sums to 1 across classes"""
            y = super().sample() + self.sample_gumbel()
            return torch.softmax(y / self.temperature, dim=-1)

        def hard_gumbel_softmax_sample(self):
            y = self.gumbel_softmax_sample()
            return (torch.max(y, dim=-1, keepdim=True)[0] == y).float()

        def rsample(self):
            return self.gumbel_softmax_sample()

        def sample(self):
            return self.rsample().detach()

        def sample_n(self, n: int) -> torch.Tensor:
            samples = torch.from_numpy(np.array([self.rsample() for _ in range(n)]).reshape(n,))
            return samples

        def hard_sample(self):
            return self.hard_gumbel_softmax_sample()

        def log_prob(self, y: torch.Tensor) -> torch.Tensor:
            loc, scale = so.fmin(lambda p, x: (-torch.log(gumbel_pdf(y, p[0], p[1]))).sum(), [0.5, 0.5],
                    args=(y,), disp = False)
           # loc, scale = torch.from_numpy(loc), torch.from_numpy(scale)
            scale = torch.from_numpy(np.array([scale])).cuda().float()
            y = (loc - y) / scale
            return (y - y.exp()) - torch.log(scale)





class SquashedGaussianDistribution(Distribution):
    _mean: torch.Tensor
    _std: torch.Tensor
    _dist: Normal

    def __init__(self, loc: torch.Tensor, std: torch.Tensor):
        self._mean = loc
        self._std = std
        self._dist = Normal(self._mean, self._std)

    def sample(self) -> torch.Tensor:
        return torch.tanh(self._dist.rsample())

    def sample_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_y = self._dist.rsample()
        log_prob = self._log_prob_from_raw_y(raw_y)
        return torch.tanh(raw_y), log_prob

    def sample_without_squash(self) -> torch.Tensor:
        return self._dist.rsample()

    def sample_n(self, n: int) -> torch.Tensor:
        return torch.tanh(self._dist.rsample((n,)))

    def sample_n_with_log_prob(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_y = self._dist.rsample((n,))
        log_prob = self._log_prob_from_raw_y(raw_y)
        return torch.tanh(raw_y), log_prob

    def sample_n_without_squash(self, n: int) -> torch.Tensor:
        return self._dist.rsample((n,))

    def mean_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tanh(self._mean), self._log_prob_from_raw_y(self._mean)

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        clipped_y = y.clamp(-0.999999, 0.999999)
        raw_y = torch.atanh(clipped_y)
        return self._log_prob_from_raw_y(raw_y)

    def _log_prob_from_raw_y(self, raw_y: torch.Tensor) -> torch.Tensor:
        jacob = 2 * (math.log(2) - raw_y - F.softplus(-2 * raw_y))
        return (self._dist.log_prob(raw_y) - jacob).sum(dim=-1, keepdims=True)

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self._mean)

    @property
    def std(self) -> torch.Tensor:
        return self._std
