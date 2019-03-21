#
# Hodgkin-Huxley toy model.
#

import numpy as np
import torch
from lfibenchmarks import NumpySimulator


class Simulator(NumpySimulator):

    """
    Toy model based on the potassium current experiments used for Hodgkin and
    Huxley's 1952 model of the action potential of a squid's giant axon.
    A voltage-step protocol is created and applied to an axon, and the elicited
    potassium current is given as model output.

    The protocol is applied in the interval ``t = [0, 1200]``, so sampling
    outside this interval will not provide much new information.

    References:

    [1] A quantitative description of membrane currents and its application to
    conduction and excitation in nerve
    Hodgkin, Huxley (1952d) Journal of Physiology
    """

    def __init__(self, initial_condition=0.3, times=np.arange(1200 * 4) / 4):
        super(Simulator, self).__init__()
        self._n0 = float(initial_condition)
        if self._n0 <= 0 or self._n0 >= 1:
            raise ValueError('Initial condition must be > 0 and < 1.')

        # Reversal potential, in mV
        self._E_k = -88

        # Maximum conductance, in mS/cm^2
        self._g_max = 36

        # Voltage step protocol
        self._prepare_protocol()
        
        self._times = torch.tensor(times, dtype=torch.float)
        if self._times[0] < 0:
            raise ValueError('Start time must be non-negative.')

    def _prepare_protocol(self):
        """
        Sets up a voltage step protocol for use with this model.
        The protocol consists of multiple steps, each starting with 90ms at a
        fixed holding potential, followed by 10ms at a varying step potential.
        """
        
        self._t_hold = 90         # 90ms at v_hold
        self._t_step = 10         # 10ms at v_step
        self._t_both = self._t_hold + self._t_step
        self._v_hold = -(0 + 75)
        self._v_step = np.array([
            -(-6 + 75),
            -(-11 + 75),
            -(-19 + 75),
            -(-26 + 75),
            -(-32 + 75),
            -(-38 + 75),
            -(-51 + 75),
            -(-63 + 75),
            -(-76 + 75),
            -(-88 + 75),
            -(-100 + 75),
            -(-109 + 75),
        ])
        self._n_steps = len(self._v_step)

        # Protocol duration
        self._duration = len(self._v_step) * (self._t_hold + self._t_step)

        # Create list of times when V changes (not including t=0)
        self._events = np.concatenate((
            self._t_both * (1 + np.arange(self._n_steps)),
            self._t_both * np.arange(self._n_steps) + self._t_hold))
        self._events.sort()

        # List of voltages (not including V(t=0))
        self._voltages = np.repeat(self._v_step, 2)
        self._voltages[1::2] = self._v_hold

    def forward(self, inputs):
        inputs = inputs.view(-1, 5)
        num_samples = inputs.size(0)

        output = torch.empty([num_samples, len(self._times)], dtype=torch.float)

        for k in range(num_samples):
            p1, p2, p3, p4, p5 = inputs[k,:]
            
            # Output arrays
            ns = torch.zeros(len(self._times))
            vs = torch.zeros(len(self._times))

            # Analytically calculate n, during a fixed-voltage step
            def calculate_n(v, n0, t0, times):
                a = p1 * (-(v + 75) + p2) / (np.exp((-(v + 75) + p2) / p3) - 1)
                b = p4 * np.exp((-v - 75) / p5)
                tau = 1 / (a + b)
                inf = a * tau
                return inf - (inf - n0) * np.exp(-(times - t0) / tau)

            # Iterate over the step, fill in the output arrays
            v = self._v_hold
            t_last = 0
            n_last = self._n0
            for i, t_next in enumerate(self._events):
                index = (t_last <= self._times) * (self._times < t_next)
                vs[index] = torch.tensor(v)
                ns[index] = calculate_n(v, n_last, t_last, self._times[index])
                n_last = calculate_n(v, n_last, t_last, t_next)
                t_last = t_next
                v = self._voltages[i]
            index = self._times >= t_next
            vs[index] = torch.tensor(v)
            ns[index] = calculate_n(v, n_last, t_last, self._times[index])
            n_last = calculate_n(v, n_last, t_last, t_next)

            # Calculate and return current
            output[k] = self._g_max * ns**4 * (vs - self._E_k)
            
        return output
"""
    def suggested_parameters(self):

        return np.array([0.01, 10, 10, 0.125, 80])

    def suggested_times(self):

        return np.arange(self._duration * 4) / 4
        
"""
