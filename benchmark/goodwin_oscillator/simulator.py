#
# Goodwin oscillator toy model.
#

import numpy as np
import torch
import scipy.integrate
from benchmark import NumpySimulator


class Simulator(NumpySimulator):

    """
    Three-state Goodwin oscillator toy model [1, 2].
    [1] Oscillatory behavior in enzymatic control processes.
    Goodwin (1965) Advances in enzyme regulation.
    [2] Mathematics of cellular control processes I. Negative feedback to one
    gene. Griffith (1968) Journal of theoretical biology.
    """

    def __init__(self, y0=[0.0054, 0.053, 1.93], times=np.linspace(0, 100, 200)):
        super(Simulator, self).__init__()
        self._y0 = torch.tensor(y0, dtype=torch.float)
        self._times = torch.tensor(times, dtype=torch.float)

    def _rhs(self, state, time, parameters):
        """
        Right-hand side equation of the ode to solve.
        """
        x, y, z = state
        k2, k3, m1, m2, m3 = parameters
        dxdt = 1 / (1 + z**10) - m1 * x
        dydt = k2 * x - m2 * y
        dzdt = k3 * y - m3 * z
        return dxdt, dydt, dzdt

    def forward(self, inputs):
        inputs = inputs.view(-1, 5)
        num_samples = inputs.size(0)

        solution = torch.empty(num_samples, len(self._times), len(self._y0))

        for i in range(num_samples):
            solution[i] = torch.tensor(scipy.integrate.odeint(self._rhs, self._y0, self._times, args=(inputs[i,0:5],)), dtype=torch.float)
            
        return solution
"""
    def suggested_parameters(self):

        return np.array([2, 4, 0.12, 0.08, 0.1])

    def suggested_times(self):

        return np.linspace(0, 100, 200)
        
"""
