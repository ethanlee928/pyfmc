import pandas as pd

from ..common import HistoricalData


def simulate(df: pd.DataFrame, open_index: str = "Open", close_index: str = "Close"):
    data = HistoricalData(df, open_index, close_index)
    return data.return_mean, data.return_std
