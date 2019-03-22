import numpy as np
import pprint

from benchmark.binomial_blob import BinomialBlob

simulator = BinomialBlob()
thetas = np.random.randn(2, 3)
outputs = simulator(thetas)

printer = pprint.PrettyPrinter(indent=4)
printer.pprint(simulator(thetas))
