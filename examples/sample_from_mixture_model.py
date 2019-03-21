import torch
import pyro
from pyro.distributions import Normal
from benchmark.mixture_model import MixtureModelSimulator

sim = MixtureModelSimulator([Normal, Normal])
theta = torch.tensor(16*[[0.5, 0.5, -1., 0.1, 1., 0.1]])
x = sim(theta)
print(x)
