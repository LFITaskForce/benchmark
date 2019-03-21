import numpy as np
import torch
import pprint

from lfibenchmarks.logistic import Simulator

inputs = torch.tensor([0.5, 100.])
simulator = Simulator()
output = simulator(inputs)

printer = pprint.PrettyPrinter(indent=4)
printer.pprint(output)
