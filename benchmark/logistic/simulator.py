#
# Logistic toy model.
#

#import numpy as np
import torch
import pyro
from benchmark import PyroSimulator


class Simulator(PyroSimulator):

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

    Has two parameters:
    r = growth rate
    k = carrying capacity

    Three hyperparameters:
    p_0 = initial population size
    noise = std. dev. of normal noise applied to output
    times = array of sampled times
    
    Has two parameters: A growth rate :math:`r` and a carrying capacity
    :math:`k`. The initial population size :math:`f(0) = p_0` can be set using
    the (optional) named constructor arg ``initial_population_size``

    Has noise drawn from an absolute valued normal distribution with mean 0 
    and std. dev.  applied as noise

    [1] https://en.wikipedia.org/wiki/Population_growth
    """

    def __init__(self, initial_population_size=2, noise=1, times=torch.linspace(0, 100, 100)):
        super(Simulator, self).__init__()
        self._p0 = float(initial_population_size)
        if self._p0 < 0:
            raise ValueError('Population size cannot be negative.')
        self._times = torch.tensor(times, dtype=torch.float)
        if torch.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        self._noise = float(noise)
        if self._noise < 0:
            raise ValueError('Noise level must be non-negative.')

    def forward(self, inputs):
        inputs = inputs.view(-1, 2)
        num_samples = inputs.size(0)
        r = inputs[:, 0]
        k = inputs[:, 1]

        values = torch.zeros(num_samples, len(self._times))
        distribution_n = pyro.distributions.Normal(0, self._noise).expand([num_samples, len(self._times)])

        if self._p0 != 0:
            ind_list = torch.nonzero(k)
            exp = torch.exp(-r[ind_list] * self._times)
            c = (k[ind_list] / self._p0 - 1)

            values[ind_list] = k[ind_list] / (1 + c * exp)

        values += torch.abs(pyro.sample("y_err", distribution_n))
        return values
"""
    def suggested_parameters(self):

        return np.array([0.1, 50])

    def suggested_times(self):

        return np.linspace(0, 100, 100)
        
"""
