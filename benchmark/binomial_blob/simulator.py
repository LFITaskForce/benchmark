import numpy as np

from benchmark import NumpySimulator


class BinomialBlob(NumpySimulator):
    """Images of blobs as in Lueckmann et al., 2018 [1]

    [1]: https://arxiv.org/abs/1805.09294
    """
    def __init__(self, img_size=32, seed=None):
        self.img_size = img_size
        self.set_seed(seed)

        self.grid_x, self.grid_y = np.meshgrid(*[np.linspace(
            -self.img_size//2, self.img_size//2, self.img_size) for _ in range(2)])

    @property
    def img_shape(self):
        return (int(self.img_size), int(self.img_size), 1)

    def set_seed(self, seed=None):
        self.random_state = np.random.RandomState(seed)

    def _get_params(self, theta):
        # Inputs: theta as 2d np.array.
        # Outputs: N, p of binomial distribution.
        assert theta.ndim == 2, 'theta must be 2d np.array'

        theta = np.atleast_2d(theta)

        xo = theta[:, 0]
        yo = theta[:, 1]
        gamma = 1.
        sigma = 2.
        if theta.shape[1] > 2:
            gamma = theta[:, 2]
        if theta.shape[1] > 3:
            sigma = theta[:, 3]

        r = (self.grid_x[:, :, None] - xo)**2 + (self.grid_y[:, :, None] - yo)**2
        p = 0.1 + 0.8 * np.exp(-0.5 * (r / sigma**2) ** gamma)
        p = p.swapaxes(1, 2).swapaxes(1, 0)

        return 255, p

    def forward(self, theta):
        """
        Forward pass of the simulator.

        Args:
            inputs (np.array): Values of the parameters with shape (n_batch, n_parameters).
            n_parameters should be at between 2 and 4:
            Parameter 1: x location of the blob
            Parameter 2: y location of the blob
            Parameter 3: contrast of the blob
            Parameter 4: scale of the blob

        Returns:
            outputs (np.array): Values of the data with shape (n_batch, self.img_size, self.img_size).
        """
        N, p = self._get_params(theta)
        return self.random_state.binomial(N, p)
