import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, t: Tensor) -> None:
        self._t = t

    def VaR(self):
        ...

    def plot_hist(self, n_bins: int, save_path: str = None):
        counts, bins = np.histogram(self._t.numpy(), bins=n_bins)
        plt.hist(bins[:-1], bins, weights=counts)
        if save_path is not None:
            plt.savefig(save_path, format="png")
