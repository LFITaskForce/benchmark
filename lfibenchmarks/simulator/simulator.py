class Simulator:
    """ Abstract simulator interface. """

    def forward(self, inputs):
        raise NotImplementedError()

    def __call__(self, inputs):
        return self.forward(inputs)
