import torch
import pyro
from pyro.distributions import Normal
from benchmark.mixture_model import MixtureModelSimulator

sim = MixtureModelSimulator([Normal, Normal])
theta = torch.tensor(16*[[0.5, 0.5, -1., 0.1, 1., 0.1]])
x = sim(theta)
log_p = sim.log_prob(theta, x)

for xi, log_pi in zip(x, log_p):
    print(xi, log_pi)
