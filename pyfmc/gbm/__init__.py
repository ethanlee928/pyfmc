import pandas as pd
from .simulation import _simulate
from ..exceptions import SimulationException


def simulate(
    df: pd.DataFrame,
    n_walkers: int,
    n_steps: int,
    step_size: float = 1.0,
    open_index: str = "Open",
    close_index: str = "Close",
    device_acc: bool = False,
):
    print(df.index)
    if (open_index and close_index) not in df.columns:
        raise SimulationException("Wrong open_index or close_index")
    return _simulate(df, n_walkers, n_steps, step_size, open_index, close_index, device_acc)
