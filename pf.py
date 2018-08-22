import toy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def gen_sample(latent_mod, obs_mod, init, size=100):
    x = np.empty(size)
    y = np.empty(size)

    x[0] = init(1)
    y[0] = obs_mod(x[0], 1)
    for i in range(1,size):
        x[i] = latent_mod(i, x[i-1], 1)
        y[i] = obs_mod(x[i], 1)

    return x, y

def particle_filter(data, prop_init, prop,
                    prop_init_lik, prop_lik, init_lik, obs_lik, trans_lik,
                    num_parts):
    size = data.size
    samples = np.empty((num_parts, size))
    weights = np.empty((num_parts, size))

    samples[:,0] = prop_init(num_parts)
    weights[:,0] = obs_lik(data[0], samples[:,0]) * init_lik(samples[:,0]) \
            / prop_init_lik(samples[:,0], data[0])

    weight_sum = np.sum(weights[:,0])
    weights[:,0] = weights[:,0]/weight_sum

    for i in range(1, size):
        # Resampling step
        selections = np.random.choice(num_parts, num_parts, p=weights[:,i-1])
        samples = samples[selections,:]
        weights[:,i-1].fill(1/num_parts)

        samples[:,i] = prop(i, samples[:,i-1], num_parts)
        weights[:,i] = weights[:,i-1] * obs_lik(data[i], samples[:,i]) \
                * trans_lik(i, samples[:,i], samples[:,i-1]) \
                / prop_lik(i, samples[:,i], samples[:,i-1], data[i])
        weight_sum = np.sum(weights[:,i])
        weights[:,i] = weights[:,i]/weight_sum

    return samples, weights

samp_size = 100
x, y = gen_sample(toy.latent, toy.obs, toy.init, size=samp_size)

# plt.figure(1)
# plt.subplot(211)
# plt.plot(range(0,samp_size), x)
#
# plt.subplot(212)
# plt.plot(range(0,samp_size), y)
# plt.show()

pf_sample, weights = particle_filter(y, toy.init, toy.latent, toy.prop_init_lik,
                            toy.prop_lik, toy.init_lik, toy.obs_lik,
                            toy.trans_lik, 1000)

print(pf_sample[:,99])
print("--------------------------")
print(weights[:,99])
# plt.plot(pf_sample[:,40], weights[:,40])
# plt.show()
