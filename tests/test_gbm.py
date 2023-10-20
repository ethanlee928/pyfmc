import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pyfmc.exceptions import SimulationException
from pyfmc.simulations.gbm import GBM


def test_sim():
    sim = GBM(
        df=pd.read_csv("./tests/data/AAPL.csv"),
        n_walkers=500_000,
        n_steps=100,
        n_trajectories=50,
        open_index="Open",
        close_index="Close",
    )
    res = sim.simulate()
    return_dist = res.return_distribution()
    return_dist.plot(kde=True)
    plt.savefig("./return_dist.png", format="png")

    price_dist = res.price_distribution()
    price_dist.plot(bins=500)
    plt.savefig("./price_dist.png", format="png")

    traj_dist = res.trajectories()
    traj_dist.plot()
    plt.savefig("./trajectory.png", format="png")


def test_wrong_df():
    with pytest.raises(SimulationException) as err:
        _ = GBM(df=pd.DataFrame({}), n_walkers=100_000, n_steps=100)
