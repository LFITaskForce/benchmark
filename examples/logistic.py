import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt

from benchmark.logistic import Simulator

times = torch.linspace(0, 100, 100)
inputs = torch.tensor([0.5, 100.])
simulator = Simulator(times = times, noise=30.)
#simulator = Simulator(sensitivities=True)
output = simulator(inputs)

printer = pprint.PrettyPrinter(indent=4)
printer.pprint(simulator.trace(inputs).nodes)

plt.scatter(times.numpy(), output.numpy(), 50)
plt.xlabel(r"$t$")
plt.ylabel(r"$f$")
plt.show()
