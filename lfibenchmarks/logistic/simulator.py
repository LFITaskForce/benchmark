#
# Logistic toy model.
#

import numpy as np
from lfibenchmarks import NumpySimulator


class Simulator(NumpySimulator):

    """
    Logistic model of population growth [1].

    .. math::
        f(t) &= \\frac{k}{1+(k/p_0 - 1)*\exp(-r t)} \\\\
        \\frac{\\partial f(t)}{\\partial r} &=
                                \\frac{k t (k / p_0 - 1) \exp(-r t)}
                                      {((k/p_0-1) \exp(-r t) + 1)^2} \\\\
        \\frac{\\partial f(t)}{ \\partial k} &= \\frac{k \exp(-r t)}
                                          {p_0 ((k/p_0-1)\exp(-r t) + 1)^2}
                                         + \\frac{1}{(k/p_0 - 1)\exp(-r t) + 1}

    Has three parameters:

    p_0 = initial population size
    r = growth rate
    k = carrying capacity
    
    Has two parameters: A growth rate :math:`r` and a carrying capacity
    :math:`k`. The initial population size :math:`f(0) = p_0` can be set using
    the (optional) named constructor arg ``initial_population_size``

    [1] https://en.wikipedia.org/wiki/Population_growth

    *Extends:* :class:`pints.Simulator`.
    """

    def __init__(self, initial_population_size=2, times=np.linspace(0, 100, 100), sensitivities=False):
        super(LogisticModel, self).__init__()
        self._p0 = float(initial_population_size)
        if self._p0 < 0:
            raise ValueError('Population size cannot be negative.')
        self._times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        self._sensitivities = sensitivities

    def forward(self, inputs):
        inputs = inputs.view(-1, 2)
        num_samples = inputs.size(0)
        r = inputs[:, 0]
        k = inputs[:, 1]

        if self._p0 == 0 or k < 0:
            if self._sensitivities:
                return np.zeros(self._times.shape), \
                    np.zeros((len(self._times), 2))
            else:
                return np.zeros(self._times.shape)

        exp = np.exp(-r * self._times)
        c = (k / self._p0 - 1)

        values = k / (1 + c * exp)

        if self._sensitivities:
            dvalues_dp = np.empty((len(self._times), 2))
            dvalues_dp[:, 0] = k * self._times * c * exp / (c * exp + 1)**2
            dvalues_dp[:, 1] = -k * exp / \
                (self._p0 * (c * exp + 1)**2) + 1 / (c * exp + 1)
            return values, dvalues_dp
        else:
            return values
"""
    def suggested_parameters(self):

        return np.array([0.1, 50])

    def suggested_times(self):

        return np.linspace(0, 100, 100)
        
"""
