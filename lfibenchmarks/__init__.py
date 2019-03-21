"""
Benchmarking problems for likelihood-free inference.
"""

__version__ = "0.0.1"

# Ordered alphabetically
__author__ = [
    "Danley Hsu",
    "Henri Pesonen",
    "Jan-Matthis Lueckmann",
    "Joeri Hermans",
    "Johann Brehmer",
    "Umberto Simola"]

# Global namespace.
from lfibenchmarks.simulator import Simulator
from lfibenchmarks.numpy_simulator import NumpySimulator
from lfibenchmarks.pyro_simulator import PyroSimulator

from pyro.util import set_rng_seed

# Default configuration.
set_rng_seed(0)
