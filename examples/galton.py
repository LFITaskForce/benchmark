import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
import sys


from benchmark.galton.simulator import Simulator

# theta = torch.tensor(5000*[[3.]])
# theta_ref = torch.tensor(5000*[[0.7]])

theta = torch.tensor(5*[[3.]])
theta_ref = torch.tensor(5*[[0.7]])


simulator = Simulator(n_rows=20, n_nails=31)
#simulator = Simulator(sensitivities=True)
# output = simulator(theta)


# print('Trace nodes =', simulator.trace(theta).nodes)
#
x, joint_score, joint_log_ratio = simulator.augmented_data(theta,theta, theta_ref)
## x, joint_score, joint_log_ratio = simulator.augmented_data(theta,None, None)

print('x = ', x)
print('joint_score = ',joint_score)
print('joint_log_ratio= ',joint_log_ratio)
print('---'*5)


#
# print(torch.mean(joint_score, dim=0))
# print(torch.mean(torch.exp(joint_log_ratio), dim=0))

#printer = pprint.PrettyPrinter(indent=4)
#printer.pprint(simulator.trace(inputs).nodes)

# plt.hist(output.numpy(), 31,range=[0,31])
# plt.xlabel(r"$x$")
# plt.ylabel("Frequency")
#
# plt.show()
