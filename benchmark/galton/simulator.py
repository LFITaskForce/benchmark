#
# Generalized Galton board example.
#
import sys
import numpy as np
import torch
from torch import nn
import pyro
from benchmark.pyro_simulator import PyroSimulator


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

    def __init__(self, n_rows=5, n_nails=31, start_pos=None):
        super(Simulator, self).__init__()

        self.n_rows = torch.tensor(n_rows, dtype=torch.float)
        if self.n_rows < 2:
            raise ValueError('Must be at least two rows.')

        self.n_nails = torch.tensor(n_nails)
        if self.n_nails < 2:
            raise ValueError('Must be at least two nails.')

        if start_pos == None:
            self.start_pos = self.n_nails // 2 # We choose the nail at the center
        else:
            self.start_pos = start_pos

        if self.start_pos < 0 or self.start_pos > self.n_nails:
            raise ValueError('Must start within bounds of [0, n_nails]')





    def nail_positions(self, theta, level=None, nail=None):

        if level is None or nail is None:
            level = torch.arange(self.n_rows, dtype=torch.float).expand(self.n_nails, self.n_rows)# With expand we get a tensor of n_nails rows and n_rows colums.
            print('level shape =',level.shape)
            level=torch.transpose(level,0,-1) #We transpose it so that each level is a row. So the 1st row has all 0, the 2nd all 1, etc

            nail = torch.arange(self.n_nails, dtype=torch.float).expand(self.n_rows, self.n_nails)# We get the full Galton board


        # level=torch.tensor(level, dtype=torch.float)
        # nail =torch.tensor(nail, dtype=torch.float)

        level_rel = 1. * level / (self.n_rows - 1)
        nail_rel = 2. * nail / (self.n_nails - 1) - 1.

        m = nn.Sigmoid()
        # sigm = m(torch.tensor(10*theta*nail_rel, dtype=torch.float))
        sigm = m(10 * theta * nail_rel)
        # print('sigm =', sigm)


        # p(z_h,z_v,theta) - see pag. 3 of 1805.12244. This gives the probability for each value [z_h,z_v] of going left. Returns 0.5 for theta=0
        nail_positions = ((1. - torch.sin(np.pi * level_rel)) * 0.5
                          + torch.sin(np.pi * level_rel) * sigm)


        return nail_positions





    def forward(self, inputs):
        inputs = inputs.view(-1, 1)
        num_samples = inputs.shape[0]
        theta = inputs[:, 0] # We could have input variables
        print('Num samples = ',num_samples)
        print('Theta = ',theta)

        # Define a pyro distribution
        # distribution_u = pyro.distributions.Uniform(0, 1).expand([num_samples])
        # dist_bern = pyro.distributions.Bernoulli(probs=a)

        # print('distribution_u = ', distribution_u)
        print('---' * 3)

        # Run and mine gold
        # left/right decisions are based on value of theta
        # log_pxz based on value of theta


        # begin = torch.zeros(num_samples).fill_(self.start_pos)
        pos = torch.zeros(num_samples).fill_(self.start_pos)

        # begin = np.empty(num_samples)
        # pos = np.empty(num_samples)

        # begin.fill(self.n_nails // 2)
        # pos.fill(self.n_nails // 2)

        # print('Begin = pos =', begin)
        print('---' * 3)

        # z = []
        z=torch.tensor(0,dtype=torch.float)

        while z < self.n_rows:
            '''
            Parametrization of the Galton Board
            
            level 0 =   0   1   2   3   ...  29  30  ...
            level 1 = 0   1   2   3   ...  29  30  ...
            level 2 =   0   1   2   3   ...  29  30  ...
            '''

            # print('pos before calling threshold = ',pos)
            # print('begin before calling threshold = ', begin)
            level = z
            print('Level = ',level)

            tmp_prob = self.nail_positions(theta, level=level, nail=pos)

            if level % 2 == 0:  # For the 1st nail (pos=0) the prob. of going left is 0 and for the last one (pos=n_nail) it is 1)
                tmp_prob[pos == float(self.n_nails)] = 1.0
            else:
                tmp_prob[pos == 0.] = 0.0

            t =tmp_prob

            # t = self.threshold(theta, (begin, z))
            # t is the probability of going left. At first z is empty so we don't change pos, and use the begin value.



            # Left indices
            dist_bern = pyro.distributions.Bernoulli(probs=t)

            # print('Sample =',dist_bern.sample())
            draw = pyro.sample("u" + str(z), dist_bern)  # We draw a number for each ball from the distribution_u


            print('t = ', t)
            print('pyro sample = ',draw) # We sample from the pyro distribution
            print('---')


            # # going left
            print('pos before = ', pos)

            # draw==1 means going left and draw==0 going right
            if level % 2 == 0:  # even rows
                pos[draw==0] = pos[draw==0] + 1 # We move to the right when ind_list[i]==0 (When the sampled value is greater than the prob. of going left).
            else:  # odd rows
                pos[draw==1] = pos[draw==1] - 1 # We move to the left when ind_list[i]==1


            print('pos after = ', pos)
            print('---' * 40)
            print('---' * 40)

            z+=1

        x = pos

        return x
"""
    def suggested_parameters(self):

        return np.array([0.1, 50])

    def suggested_times(self):

        return np.linspace(0, 100, 100)
        
"""
