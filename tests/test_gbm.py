import pandas as pd
from pyfmc import gbm
import numpy as np
import matplotlib.pyplot as plt


def test_sim():
    dist = gbm.simulate(df=pd.read_csv("./tests/data/AAPL.csv"), n_walkers=100_000, n_steps=100)
    counts, bins = np.histogram(dist.numpy(), bins=500)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig("./result.png", format="png")
