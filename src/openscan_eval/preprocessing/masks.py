from __future__ import annotations

import cv2
import numpy as np


def postprocess_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    for operation, key in ((cv2.MORPH_OPEN, "opening_kernel"), (cv2.MORPH_CLOSE, "closing_kernel")):
        size = int(config.get(key, 0))
        if size > 1:
            binary = cv2.morphologyEx(binary, operation, np.ones((size, size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        minimum = binary.size * float(config.get("min_component_ratio", 0))
        keep = [i + 1 for i, area in enumerate(areas) if area >= minimum]
        if config.get("keep_largest_component", True) and keep:
            keep = [max(keep, key=lambda i: stats[i, cv2.CC_STAT_AREA])]
        binary = np.isin(labels, keep).astype(np.uint8) * 255
    if config.get("fill_holes", False):
        flood = binary.copy(); padded = np.zeros((binary.shape[0]+2, binary.shape[1]+2), np.uint8)
        cv2.floodFill(flood, padded, (0, 0), 255)
        binary |= cv2.bitwise_not(flood)
    return binary


def build_background(images: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(images), axis=0).astype(np.uint8)


def foreground_mask(image: np.ndarray, background: np.ndarray, config: dict) -> np.ndarray:
    if config.get("mode") == "automatic":
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        mask = (gradient >= float(config.get("gradient_threshold", 20))).astype(np.uint8) * 255
        x0, y0, x1, y1 = config.get("scan_roi_xyxy", [.33, .30, .60, .82])
        roi = np.zeros_like(mask); roi[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = 255; mask &= roi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = (hsv[:,:,0] >= 85) & (hsv[:,:,0] <= 130) & (hsv[:,:,1] >= int(config.get("blue_saturation_min",120)))
        mask[blue] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25,25),np.uint8))
        mask = cv2.dilate(mask, np.ones((5,5),np.uint8))
        base_roi = np.zeros_like(mask)
        cv2.ellipse(base_roi, (int(.45*w), int(.68*h)), (int(.09*w), int(.065*h)), 0, 0, 360, 255, -1)
        channel_spread = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
        neutral_metal = ((channel_spread < 35) & (image.mean(axis=2) > 65)).astype(np.uint8) * 255
        neutral_metal &= base_roi
        neutral_metal = cv2.morphologyEx(neutral_metal, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
        mask |= neutral_metal
        mask[blue] = 0
        return postprocess_mask(mask, config)
    diff = cv2.absdiff(image, background)
    score = np.max(diff, axis=2)
    mask = (score >= int(config["difference_threshold"])).astype(np.uint8) * 255
    # Exclude border-connected response and favor the scan volume near image center.
    border = int(min(mask.shape) * .02)
    mask[:border] = 0; mask[-border:] = 0; mask[:, :border] = 0; mask[:, -border:] = 0
    return postprocess_mask(mask, config)
