import pyro
import pprint
import torch

from benchmark.glm_regression import GLMRegression as Simulator



simulator = Simulator()
thetas = torch.randn(10, 3)
outputs = simulator(thetas)

printer = pprint.PrettyPrinter(indent=4)
printer.pprint(simulator.trace(thetas).nodes)
