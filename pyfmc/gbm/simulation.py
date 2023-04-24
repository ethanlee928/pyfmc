import logging
from math import sqrt

import torch
import pandas as pd
from tqdm import trange

from ..common import HistoricalData, get_device

logger = logging.getLogger("pyfmc.gbm")


def _simulate(
    df: pd.DataFrame,
    n_walkers: int,
    n_steps: int,
    step_size: float,
    open_index: str,
    close_index: str,
    device_acc: bool,
):
    hist_data = HistoricalData(df, open_index, close_index)
    device = get_device() if device_acc else torch.device("cpu")
    logger.info("Using device: %s", device)

    exp_return = torch.tensor(hist_data.return_mean, device=device)
    std_return = torch.tensor(hist_data.return_std, device=device)
    last_price = hist_data.get_latest_close_price()

    s0 = torch.tensor([last_price for _ in range(n_walkers)], device=device)
    s1 = torch.zeros(n_walkers, device=device)
    ds = torch.zeros(n_walkers, device=device)
    dt = torch.tensor(step_size)

    for _ in trange(n_steps):
        epsilon = torch.randn(n_walkers, device=device)
        shock = torch.multiply(std_return * sqrt(dt), epsilon)
        drift = torch.multiply(dt, exp_return)
        _sum = torch.add(shock, drift)
        ds = torch.multiply(_sum, s0)
        s1 = torch.add(s0, ds)
        s0 = torch.clone(s1)

    s0 = s0[torch.isfinite(s0)]
    return s0.cpu()
