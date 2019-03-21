"""Generalized Linear Model

Probabilistic generative linear model.
"""

import torch

from lfibenchmarks.simulator import PyroSimulator



class Simulator(PyroSimulator):

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError
