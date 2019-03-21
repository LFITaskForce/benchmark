import torch

import pyro
import pyro.distributions as dist


def simulator(parameters):
    return parameters

input = torch.tensor([1,2])
trace = pyro.poutine.trace(simulator).get_trace(input)

print(trace)
