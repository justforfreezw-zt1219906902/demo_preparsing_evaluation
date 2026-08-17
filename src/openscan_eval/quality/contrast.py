import numpy as np


def contrast_metric(gray: np.ndarray) -> float:
    return float(np.std(gray))
