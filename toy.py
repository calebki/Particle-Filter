#Simple nonlinear time series model
import numpy as np
import math
global sigma1
sigma1 = math.sqrt(10)
global sigma2
sigma2 = 1

def latent(t, prev, n):
    curr = prev/2 + 25*prev/(1+prev**2) + 8*math.cos(1.2*t) \
         + np.random.normal(scale=sigma1, size = n)
    return curr

def obs(x, n):
    y = x**2/20 + np.random.normal(scale=sigma2, size = n)
    return y

def init(n):
    return np.random.normal(scale=math.sqrt(10), size = n)

def init_lik(x):
    lik = 1/((2*math.pi)*10)**(1/2) * np.exp(
    -1* x**2 / (20))
    return lik

def obs_lik(y, x):
    lik = 1/((2*math.pi)*sigma2)**(1/2) * np.exp(
    -1*(y-x**2/20)**2 / (2*sigma2**2))
    return lik

def trans_lik(t, x, prev):
    lik = 1/(2*math.pi*sigma1)**(1/2) * np.exp(
    -1*(x-(prev/2 + 25*prev/(1+prev**2) + 8*math.cos(1.2*t)))**2 / (2*sigma1**2))
    return lik

def prop_init_lik(x ,y):
    return init_lik(x)

def prop_lik(t, x, prev, y):
    return trans_lik(t, x, prev)
