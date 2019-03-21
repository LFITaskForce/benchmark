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

# Global namespace
from benchmark.simulator import Simulator
from benchmark.numpy_simulator import NumpySimulator
from benchmark.pyro_simulator import PyroSimulator

from pyro.util import set_rng_seed
