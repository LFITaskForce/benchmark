import torch
from pyro.distributions import Normal
from benchmark.mixture_model import MixtureModelSimulator

sim = MixtureModelSimulator([Normal, Normal])

n_samples = 1000
theta = torch.tensor([[0.3, 0.7, -0.5, 0.1, 1., 0.2]])
theta_reference = torch.tensor([[0.5, 0.5, -1., 1., 1., 1.]])

xs = []
joint_scores = []
joint_log_likelihood_ratios = []

for _ in range(n_samples):
    x, joint_score, joint_log_ratio = sim.augmented_data(theta, theta, theta_reference)
    theta.requires_grad = True
    trace = sim.trace(theta)
    xs.append(x)
    joint_scores.append(joint_score)
    joint_log_likelihood_ratios.append(joint_log_ratio)

print(xs)
print(joint_scores)
print(joint_log_likelihood_ratios)

print(torch.mean(joint_scores))
print(torch.mean(torch.exp(-joint_log_likelihood_ratios)))
