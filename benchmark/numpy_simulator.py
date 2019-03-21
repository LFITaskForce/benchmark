from benchmark import Simulator



class NumpySimulator(Simulator):
    """ Numpy simulator interface """

    def forward(self, inputs):
        raise NotImplementedError
