import pyro

from benchmark.simulator import Simulator



class PyroSimulator(Simulator):
    """ Pyro simulator interface """

    def forward(self, inputs):
        raise NotImplementedError()

    def trace(self, inputs):
        return pyro.poutine.trace(self.forward).get_trace(inputs)
