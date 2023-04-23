import logging
import torch

from .historical_data import HistoricalData

logger = logging.getLogger("pyfmc.common")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("CUDA or MPS not available, using CPU for calculations ...")
    return torch.device("cpu")
