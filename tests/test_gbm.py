import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyfmc.simulations.gbm import GBM
from pyfmc.exceptions import SimulationException


def test_sim():
    sim = GBM(df=pd.read_csv("./tests/data/AAPL.csv"), n_walkers=100_000, n_steps=100)
    res = sim.simulate()
    return_dist = res.return_distribution()
    counts, bins = np.histogram(return_dist, bins=500, density=True)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title("Test GBM")
    plt.savefig("./result.png", format="png")


def test_wrong_df():
    with pytest.raises(SimulationException) as err:
        _ = GBM(df=pd.DataFrame({}), n_walkers=100_000, n_steps=100)
