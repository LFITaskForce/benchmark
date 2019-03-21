"""Generalized Linear Model

Probabilistic generative linear model.
"""

import torch

from lfibenchmarks import PyroSimulator



class Simulator(PyroSimulator):

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, inputs):
        inputs = inputs.view(-1, 3)
        num_samples = inputs.size(0)
        # Fetch the seperate parameters.
        distribution = pyro.distributions.Uniform(0, 1).expand(num_samples)
        m = inputs[:, 0]
        b = inputs[:, 1]
        f = inputs[:, 2]
        x = 10 * pyro.sample("x", distribution)
        y_err = .1 + .5 * pyro.sample("y_err", distribution)
        y = m * x + b
        y += (f * y).abs() * pyro.sample("y_1", distribution)
        y += y_err * pyro.sample("y_2", distribution)
        outputs = torch.cat([x, y, y_err], dim=1)

        return outputs
