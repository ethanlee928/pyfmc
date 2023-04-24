import logging
from math import sqrt

import torch
import pandas as pd
from tqdm import trange

from . import Simulations
from ..exceptions import SimulationException
from ..common import HistoricalData, get_device, Distribution

logger = logging.getLogger("pyfmc.simulations.gbm")


class GBM(Simulations):
    def __init__(
        self,
        df: pd.DataFrame,
        n_walkers: int,
        n_steps: int,
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
        if (open_index and close_index) not in df.columns:
            raise SimulationException("Wrong open_index or close_index")

    def simulate(self):
        hist_data = HistoricalData(self.df, self.open_index, self.close_index)
        device = get_device() if self.device_acc else torch.device("cpu")
        logger.info("Using device: %s", device)

        exp_return = torch.tensor(hist_data.return_mean, device=device)
        std_return = torch.tensor(hist_data.return_std, device=device)
        last_price = hist_data.get_latest_close_price()

        s0 = torch.tensor([last_price for _ in range(self.n_walkers)], device=device)
        s1 = torch.zeros(self.n_walkers, device=device)
        ds = torch.zeros(self.n_walkers, device=device)
        dt = torch.tensor(self.step_size)

        for _ in trange(self.n_steps):
            epsilon = torch.randn(self.n_walkers, device=device)
            shock = torch.multiply(std_return * sqrt(dt), epsilon)
            drift = torch.multiply(dt, exp_return)
            _sum = torch.add(shock, drift)
            ds = torch.multiply(_sum, s0)
            s1 = torch.add(s0, ds)
            s0 = torch.clone(s1)

        s0 = s0[torch.isfinite(s0)]
        return Distribution(s0.cpu())
