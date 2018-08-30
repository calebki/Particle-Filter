import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from toy import Example1


def gen_sample(size=100):
    ex = Example1(1)
    return np.array(list(zip(*[ex.sample() for _ in range(size)])))


def particle_filter(data, num_parts):
    size = data.size
    ex = Example1(num_parts)
    samples = np.zeros((size, num_parts))
    weights = np.zeros_like(samples)
    samples[0] = ex.xt
    weights[0] = (
        ex.obs_dist(xt=samples[0]).pdf(x=data[0])
        # * ex.init_dist.pdf(x=samples[0])
        # / ex.init_dist.pdf(x=samples[0])
    )
    weights[0] /= weights[0].sum()

    for t in range(1, size):
        # Resampling step
        samples[t] = np.random.choice(
            samples[t - 1],
            num_parts, 
            p=weights[t - 1], 
            replace=True
        )  # resample x_{t-1} and temporarily store as x_t
        samples[t] = ex.latent_dist(t, samples[t]).rvs()
        weights[t] = ex.obs_dist(samples[t]).pdf(data[t])
        weights[t] /= weights[t].sum()

    return samples, weights


samp_size = 100
x, y = gen_sample(samp_size)

num_size = 100000
pf_sample, weights = particle_filter(y, num_size)

plt.scatter(pf_sample[40], weights[40])
plt.show()
