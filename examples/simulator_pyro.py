import torch
import pyro
import pyro.distributions as dist


def simulator(inputs):
    w = torch.tensor([0.5, 0.5])
    n = pyro.sample("noise", dist.Normal(0., 1.0))
    return torch.dot(inputs, w) + n


inputs = torch.tensor([1., 2.])
trace = pyro.poutine.trace(simulator).get_trace(inputs)

print(dir(trace))
print(trace.nodes)
