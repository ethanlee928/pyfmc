import pandas as pd


class HistoricalData:
    def __init__(self, raw_data: pd.DataFrame, open_index: str = "Open", close_index: str = "Close") -> None:
        self._raw_data = raw_data
        self.open_index = open_index
        self.close_index = close_index

        self._return_mean: float = None
        self._return_std: float = None

    @property
    def return_mean(self):
        if self._return_mean is None:
            self.mean_and_std()
        return self._return_mean

    @property
    def return_std(self):
        if self._return_std is None:
            self.mean_and_std()
        return self._return_std

    def mean_and_std(self):
        _returns = (self._raw_data[self.close_index] - self._raw_data[self.open_index]) / self._raw_data[
            self.open_index
        ]
        self._return_mean = _returns.mean()
        self._return_std = _returns.std()

    def get_latest_close_price(self) -> float:
        return self._raw_data[self.close_index].iloc[-1]
