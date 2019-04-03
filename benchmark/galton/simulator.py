#
# Generalized Galton board example.
#

import numpy as np
import torch
import pyro
from benchmark import PyroSimulator


class Simulator(PyroSimulator):

    """
    Generalized Galton board example from arXiv:1805.12244.


    Has one parameter:
    theta

    Three hyperparameters:
    n_row = number of rows
    n_nails = number of nails
    start_pos = starting position (default = n_nails / 2)
    """

    def __init__(self, n_rows=20, n_nails=31, start_pos=None):
        super(Simulator, self).__init__()
        self.n_rows = torch.tensor(n_rows)
        if self.n_rows < 2:
            raise ValueError('Must be at least two rows.')
        self.n_nails = torch.tensor(n_nails)
        if self.n_nails < 2:
            raise ValueError('Must be at least two nails.')
        if start_pos == None:
            self.start_pos = self.n_nails // 2
        else:
            self.start_pos = start_pos
        if self.start_pos < 0 or self.start_pos > self.n_nails:
            raise ValueError('Must start within bounds of [0, n_nails]')

    def nail_positions(self, theta, level=None, nail=None):
        if level is None or nail is None:
            level = torch.arange(self.n_rows, dtype=torch.float).expand(self.n_nails, self.n_rows)
            nail = torch.arange(self.n_nails, dtype=torch.float).expand(self.n_rows, self.n_nails)

        level_rel = 1. * torch.tensor(level, dtype=torch.float) / (self.n_rows - 1)
        nail_rel = 2. * nail / (self.n_nails - 1) - 1.

        sigm = 1. / (1. + torch.exp(-(10*theta*nail_rel)))
        nail_positions = ((1. - torch.sin(np.pi * level_rel)) * 0.5
                          + torch.sin(np.pi * level_rel) * sigm)

        return nail_positions

    def threshold(self, theta, trace):
        begin, z = trace
        pos = begin
        level = 0
        for step in z:
            if level % 2 == 0:
                pos[step==0] = pos[step==0]
                pos[step!=0] = pos[step!=0] + 1
            else:
                pos[step==0] = pos[step==0] - 1
                pos[step!=0] = pos[step!=0]

            level += 1

        tmp_nail = self.nail_positions(theta, level, pos)

        if level % 2 == 1:  # for odd rows, the first and last nails are constant
            tmp_nail[pos == 0.] = 0.0
            tmp_nail[pos == float(self.n_nails)] = 1.0

        return tmp_nail

    def forward(self, inputs):
        inputs = inputs.view(-1, 1)
        num_samples = inputs.size(0)
        theta = inputs[:, 0]

        distribution_u = pyro.distributions.Uniform(0, 1).expand([num_samples])

        # Run and mine gold
        # left/right decisions are based on value of theta
        # log_pxz based on value of theta

        begin = pos = torch.zeros(num_samples).fill_(self.start_pos)
        z = []

        while len(z) < self.n_rows:
            t = self.threshold(theta, (begin, z))
            level = len(z)

            # Left indicies
            ind_list = (pyro.sample("u"+str(len(z)), distribution_u) < t) | (t == 1.0)

            # going left
            if level % 2 == 0:  # even rows
                pos[ind_list] = pos[ind_list]
                pos[~ind_list] = pos[~ind_list] + 1
            else:  # odd rows
                pos[ind_list] = pos[ind_list] - 1
                pos[~ind_list] = pos[~ind_list]

            tmp_z = torch.zeros(num_samples)
            tmp_z[~ind_list] = 1
            z.append(tmp_z)
            
        x = pos

        return x
"""
    def suggested_parameters(self):

        return np.array([0.1, 50])

    def suggested_times(self):

        return np.linspace(0, 100, 100)
        
"""
