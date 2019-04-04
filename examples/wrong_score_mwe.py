import torch
from torch.autograd import grad
import pyro
from pyro.distributions import Normal

n_samples = 4
mean = torch.tensor([1. for _ in range(n_samples)], requires_grad=True)
std = torch.tensor([2. for _ in range(n_samples)], requires_grad=True)

normal = Normal(mean, std)
x = pyro.sample("normal", normal)

log_prob = normal.log_prob(x)

score_mean = grad(
    log_prob,
    mean,
    grad_outputs=torch.ones_like(log_prob.data),
    only_inputs=True,
    retain_graph=True,
)[0]

score_std = grad(
    log_prob,
    std,
    grad_outputs=torch.ones_like(log_prob.data),
    only_inputs=True,
)[0]

true_score_mean = (x - mean) / std**2
true_score_std = (x - mean)**2 / std**3 - 1. / std

print("x: {}".format(x.detach().numpy()))
print("score wrt mean: {},\n        should be {}".format(score_mean.detach().numpy(), true_score_mean.detach().numpy()))
print("score wrt std: {},\n        should be {}".format(score_std.detach().numpy(), true_score_std.detach().numpy()))
