import numpy as np
import torch
import pprint

from benchmark.logistic import Simulator

inputs = torch.tensor([0.5, 100.])
simulator = Simulator()
#simulator = Simulator(sensitivities=True)
output = simulator(inputs)

printer = pprint.PrettyPrinter(indent=4)
printer.pprint(output)
