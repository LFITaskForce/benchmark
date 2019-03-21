import numpy as np
import torch
import pprint

from lfibenchmarks.goodwin_oscillator import Simulator

inputs = torch.tensor([2, 4, 0.12, 0.08, 0.1])
simulator = Simulator()
#simulator = Simulator(sensitivities=True)
output = simulator(inputs)

printer = pprint.PrettyPrinter(indent=4)
printer.pprint(output)
