import cv2
import numpy as np


def sharpness_metrics(gray: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    pixels = mask > 0 if mask is not None and np.any(mask) else np.ones(gray.shape, bool)
    return float(np.var(lap[pixels])), float(np.mean(gx[pixels] ** 2 + gy[pixels] ** 2))
