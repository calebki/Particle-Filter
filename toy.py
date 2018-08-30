# Simple nonlinear time series model
import numpy as np
import math
import scipy.stats

SIGMA_U = math.sqrt(10)
SIGMA_V = 1


class Example1:

    def __init__(self, n):
        self.xt = self.initial_dist().rvs(size=n)
        self._t = 1

    def initial_dist(self):
        return scipy.stats.norm(loc=0, scale=SIGMA_V)

    def latent_dist(self, t, xt1):
        "Distribution of x_t conditional on x_{t-1}, t"
        mu = xt1 / 2 + 25 * xt1 / (1 + xt1 ** 2) + 8 * math.cos(1.2 * t)
        return scipy.stats.norm(loc=mu, scale=SIGMA_U)

    def obs_dist(self, xt):
        return scipy.stats.norm(loc=xt ** 2 / 20, scale=SIGMA_V)

    def sample(self):
        "Given current state x_{t-1}, sample a pair (x_t, y_t)"
        self.xt = self.latent_dist(self._t, self.xt).rvs()
        yt = self.obs_dist(self.xt).rvs()
        self._t += 1
        return np.array((self.xt, yt))
