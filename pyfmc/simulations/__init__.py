from abc import ABC


class Simulations(ABC):
    def __init__(self, n_walkers: int, n_steps: int, device_acc: bool) -> None:
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.device_acc = device_acc

    def simulate(self):
        ...
