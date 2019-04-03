import torch
from pyro.distributions import Normal
from benchmark.mixture_model import MixtureModelSimulator
from benchmark.analyse_trace import calculate_x, calculate_joint_score, calculate_joint_likelihood_ratio

sim = MixtureModelSimulator([Normal, Normal])

n_samples = 1
theta = torch.tensor([[0.5, 0.5, -1., 0.1, 1., 0.1]])
theta_reference = torch.tensor([[0.5, 0.5, -1., 0.1, 1., 0.1]])

x = []
joint_scores = []
joint_likelihood_ratios = []

for _ in range(n_samples):
    theta.requires_grad = True
    trace = sim.trace(theta)
    x.append(
        calculate_x(trace)
    )
    joint_scores.append(
        calculate_joint_score(trace, theta)
    )
    joint_likelihood_ratios.append(
        calculate_joint_likelihood_ratio(trace, theta, theta_reference)
    )

print(x)
print(joint_scores)
print(joint_likelihood_ratios)
