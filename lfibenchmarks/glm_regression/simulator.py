"""Generalized Linear Model

Probabilistic generative linear model.
"""

import torch

from lfibenchmarks.simulator import PyroSimulator



class Simulator(PyroSimulator):

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, inputs):
        inputs = inputs.view(-1, 3)
        num_samples = inputs.size(0)
        # Fetch the seperate parameters.
        m = inputs[:, 0]
        b = inputs[:, 1]
        f = inputs[:, 2]
        x = 10 * torch.rand(num_samples)
        y_err = .1 + .5 * torch.rand(num_samples)
        y = m * x + b
        y += (f * y).abs() * torch.rand(num_samples)
        y += y_err * torch.rand(num_samples)
        outputs = torch.cat([x, y, y_err], dim=1)

        return outputs
