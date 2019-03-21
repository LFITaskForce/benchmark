import torch
import pyro

from benchmark import PyroSimulator


class MixtureModelSimulator(PyroSimulator):
    """
    Simulator for a general mixture model.

    Args:
        distributions (list of pyro.distributions.Distribution): Distributions for the individual components of the
        mixture model.

    """

    def __init__(self, distributions):
        super(MixtureModelSimulator, self).__init__()

        self.distributions = distributions
        for distribution in distributions:
            if not (isinstance(distribution, pyro.distributions.Distribution)
                    or isinstance(distribution, torch.distributions.Distribution)):
                raise ValueError("Distribution is not a pyro distribution!")

        self.n_components = len(self.distributions)
        self.n_component_params = [len(dist.arg_constraints) for dist in distributions]
        self.n_params = self.n_components + sum(self.n_component_params)

    def forward(self, inputs):
        """
        Forward pass of the simulator.

        Args:
            inputs (torch.Tensor): Values of the parameters with shape (n_batch, n_parameters). The first n columns
            (where n is the number of components of the mixture model) define the weights for each component. These
            weights have to be non-negative, but do not necessarily have to sum to one (they will be rescaled to sum
            one). The following columns of inputs define the parameters to all individual distributions. FOr instance,
            in a mixture of two Gaussians, the first two parameter columns will define the two weights, the third will
            be the mean of the first Gaussian, the fourth the standard deviation of the first Gaussian, the fifth the
            mean of the second Gaussian, and the last one the standard deviation of the second Gaussian.

        Returns:
            outputs (torch.Tensor): Values of the data with shape (n_batch, 1).

        """
        assert inputs.size[1] == self.n_params, "Inconsistent shape"

        weights = inputs[:, :self.n_components]
        weights /= torch.sum(weights, axis=1)

        x = 0.
        n_params_previous = self.n_components

        for i, (weight, dist, n_component_params) in enumerate(
                zip(weights, self.distributions, self.n_component_params)
        ):
            component_params = inputs[:, n_params_previous:n_params_previous + n_component_params]
            component_params = [component_params[:, i] for i in range(n_component_params)]

            x_component = pyro.sample("component_{}".format(i), dist(*component_params))

            x = x + weight * x_component
            n_params_previous += n_component_params

        return x
