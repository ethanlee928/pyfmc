import pytest
import pandas as pd

from pyfmc.simulations.gbm import GBM
from pyfmc.exceptions import SimulationException


def test_sim():
    sim = GBM(df=pd.read_csv("./tests/data/AAPL.csv"), n_walkers=100_000, n_steps=100)
    dist = sim.simulate()
    dist.plot_hist(500, "./result.png")


def test_wrong_df():
    with pytest.raises(SimulationException) as err:
        _ = GBM(df=pd.DataFrame({}), n_walkers=100_000, n_steps=100)
