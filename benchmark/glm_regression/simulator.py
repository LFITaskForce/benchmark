"""Generalized Linear Model

Probabilistic generative linear model.
"""

import torch
import pyro

from benchmark import PyroSimulator



class GLMRegression(PyroSimulator):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        inputs = inputs.view(-1, 3)
        num_samples = inputs.size(0)
        # Fetch the seperate parameters.
        distribution_u = pyro.distributions.Uniform(0, 1).expand([num_samples])
        distribution_n = pyro.distributions.Normal(0, 1).expand([num_samples])
        m = inputs[:, 0]
        b = inputs[:, 1]
        f = inputs[:, 2]
        x = 10 * pyro.sample("x", distribution_u)
        y_err = .1 + .5 * pyro.sample("y_err", distribution_u)
        y = m * x + b
        y = y + (f * y).abs() * pyro.sample("y_1", distribution_n)
        y = y + y_err * pyro.sample("y_2", distribution_n)
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        y_err = y_err.view(-1, 1)
        outputs = torch.cat([x, y, y_err], dim=1)

        return outputs
