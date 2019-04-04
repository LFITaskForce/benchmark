import torch
import pyro
from pyro.distributions import Normal
from benchmark.pyro_simulator import PyroSimulator


class SuperSimpleSimulator(PyroSimulator):
    def forward(self, inputs):
        n_samples = inputs.shape[0]
        z0 = pyro.sample("z0", Normal(inputs[:, 0], inputs[:, 1]))
        z1 = pyro.sample("z1", Normal(torch.zeros(n_samples), torch.ones(n_samples)))
        return z0 + z1


sim = SuperSimpleSimulator()

n_samples = 10000
theta = torch.tensor([[0.3, 1.5] for _ in range(n_samples)])
theta_ref = torch.tensor([[0., 1.] for _ in range(n_samples)])

x, joint_score, joint_log_ratio = sim.augmented_data(theta, theta_ref, None)

print(x)  # Seems okay
print(joint_score)  # The first component is always 0. It seems that the mean of the Gaussian does not affect log_prob?
print(joint_log_ratio)  # Seems to work

print(torch.mean(joint_score, dim=0))  # The second component does not average out to 0, as it should...
print(torch.mean(torch.exp(joint_log_ratio), dim=0))  # Seems to work
