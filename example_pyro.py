import torch
import pyro
import pyro.distributions as dist


def simulator(parameters):
    w = torch.tensor([0.5, 0.5])
    n = pyro.sample("noise", dist.Normal(0., 1.0))
    return torch.dot(parameters, w) + n


input = torch.tensor([1., 2.])
trace = pyro.poutine.trace(simulator).get_trace(input)

print(trace.nodes)
