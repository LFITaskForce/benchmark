import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt

from benchmark.galton import Simulator

inputs = torch.tensor(5000*[[3.]])
simulator = Simulator()
#simulator = Simulator(sensitivities=True)
output = simulator(inputs)

#printer = pprint.PrettyPrinter(indent=4)
#printer.pprint(simulator.trace(inputs).nodes)

plt.hist(output.numpy(), 31)
plt.xlabel(r"$x$")
plt.ylabel("Frequency")
plt.show()
