import numpy as np
import torch
import pprint

from benchmark.hodgkin_huxley import Simulator

inputs = torch.tensor([0.01, 10, 10, 0.125, 80])
simulator = Simulator()
#simulator = Simulator(sensitivities=True)
output = simulator(inputs)

printer = pprint.PrettyPrinter(indent=4)
printer.pprint(output)
