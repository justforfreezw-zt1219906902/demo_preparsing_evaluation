import numpy as np


def exposure_metrics(gray: np.ndarray, dark: int = 5, bright: int = 250) -> tuple[float, float, float]:
    return float(gray.mean()), float((gray <= dark).mean() * 100), float((gray >= bright).mean() * 100)
