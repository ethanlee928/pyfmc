import logging
from math import sqrt
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import trange

from ..common import HistoricalData, get_device
from ..exceptions import SimulationException
from . import Simulations

logger = logging.getLogger("pyfmc.simulations.gbm")


class Trajectory:
    def __init__(self, dist: torch.Tensor, label: str = "Trajectory") -> None:
        self.dist = dist.numpy()
        self.label = label

    def value(self):
        return self.dist

    def plot(self, title=None, xlabel=None, ylabel=None):
        fig, ax = plt.subplots()
        sns.lineplot(data=self.dist, legend=False)
        ax.set_xlabel(xlabel or "time")
        ax.set_ylabel(ylabel or self.label)
        ax.set_title(title or self.label)
        return fig, ax


class Distribution:
    def __init__(self, dist: torch.Tensor, label: str = "Distribution"):
        self.dist = dist.numpy()
        self.label = label

    def value(self):
        return self.dist

    def plot(self, bins=10, kde=False, title=None, xlabel=None, ylabel=None):
        fig, ax = plt.subplots()
        if kde:
            sns.kdeplot(data=self.dist, color="blue", fill=True, ax=ax)
        else:
            sns.histplot(self.dist, bins=bins, ax=ax)
        ax.set_xlabel(xlabel or self.label)
        ax.set_ylabel(ylabel or ("Density" if kde else "Counts"))
        ax.set_title(title or self.label)
        return fig, ax

    def __str__(self) -> str:
        return str(self.dist)


class GBMResult:
    def __init__(
        self, init_dist: torch.Tensor, final_dist: torch.Tensor, trajectories: Optional[torch.Tensor] = None
    ) -> None:
        self.init_dist = init_dist.cpu()
        self.final_dist = final_dist.cpu()
        self._trajectories = trajectories.cpu() if trajectories is not None else trajectories

    def price_distribution(self):
        return Distribution(self.final_dist, label="Price Distribution")

    def trajectories(self):
        if self._trajectories is None:
            logger.warning("No trajectories")
            return
        return Trajectory(self._trajectories)

    def return_distribution(self):
        return Distribution(
            torch.div(torch.sub(self.final_dist, self.init_dist), self.init_dist), label="Return Distribution"
        )

    def VaR(self, alpha: float):
        return np.percentile(self.return_distribution().value(), alpha)


class GBM(Simulations):
    def __init__(
        self,
        df: pd.DataFrame,
        n_walkers: int,
        n_steps: int,
        n_trajectories: int = 0,
        step_size: float = 1,
        open_index: str = "Open",
        close_index: str = "Close",
        device_acc: bool = False,
    ) -> None:
        super().__init__(n_walkers, n_steps, device_acc)
        self.df = df
        self.step_size = step_size
        self.open_index = open_index
        self.close_index = close_index
        self.n_trajectories = n_trajectories
        if (open_index and close_index) not in df.columns:
            raise SimulationException("Wrong open_index or close_index")

    def simulate(self):
        hist_data = HistoricalData(self.df, self.open_index, self.close_index)
        device = get_device() if self.device_acc else torch.device("cpu")
        logger.info("Using %s for calculation ...", device)
        dtype = torch.float32 if device == torch.device("mps") else torch.float64
        logger.info("Using device: %s", device)

        exp_return = torch.tensor(hist_data.return_mean, device=device, dtype=dtype)
        std_return = torch.tensor(hist_data.return_std, device=device, dtype=dtype)
        last_price = hist_data.get_latest_close_price()

        s0 = torch.tensor([last_price] * self.n_walkers, device=device, dtype=dtype)
        init_dist = torch.clone(s0)
        trajectories = init_dist[: self.n_trajectories] if self.n_trajectories > 0 else None

        s1 = torch.zeros(self.n_walkers, device=device, dtype=dtype)
        ds = torch.zeros(self.n_walkers, device=device, dtype=dtype)
        dt = torch.tensor(self.step_size)

        for _ in trange(self.n_steps):
            epsilon = torch.randn(self.n_walkers, device=device, dtype=dtype)
            shock = (std_return * sqrt(dt)) * epsilon
            drift = dt * exp_return
            ds = (shock + drift) * s0
            s1 = s0 + ds
            s0 = torch.clone(s1)
            if self.n_trajectories > 0:
                trajectories = torch.concat((trajectories, s0[: self.n_trajectories]))

        s0 = s0[torch.isfinite(s0)]
        return GBMResult(
            init_dist,
            s0,
            trajectories.reshape([self.n_steps + 1, self.n_trajectories]) if self.n_trajectories > 0 else None,
        )
