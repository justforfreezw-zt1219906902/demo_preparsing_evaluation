from __future__ import annotations

import cv2
import numpy as np


def cleanup_probability(probability: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively clean a binary support mask while retaining soft alpha."""
    probability = np.clip(probability.astype(np.float32), 0, 1)
    binary = (probability >= float(config.get("alpha_threshold", .5))).astype(np.uint8) * 255
    for operation, key in ((cv2.MORPH_OPEN,"opening_kernel"),(cv2.MORPH_CLOSE,"closing_kernel")):
        size = int(config.get(key, 0))
        if size > 1:
            if size % 2 == 0: size += 1
            binary = cv2.morphologyEx(binary, operation, np.ones((size,size),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    keep=[]; minimum=binary.size*float(config.get("min_component_ratio",0))
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA] >= minimum: keep.append(i)
    if config.get("keep_largest_component",False) and keep:
        keep=[max(keep,key=lambda i:stats[i,cv2.CC_STAT_AREA])]
    support=np.isin(labels,keep).astype(np.uint8) if keep else np.zeros_like(binary)
    soft=probability*support
    radius=float(config.get("feather_radius",0))
    if radius>0: soft=cv2.GaussianBlur(soft,(0,0),radius)
    return np.clip(soft,0,1),support*255


def background_model(images: list[np.ndarray]) -> np.ndarray:
    if not images: raise ValueError("At least one image is required for background subtraction")
    return np.median(np.stack(images),axis=0).astype(np.uint8)


def background_probability(image: np.ndarray, background: np.ndarray, threshold: float) -> np.ndarray:
    difference=cv2.absdiff(image,background).max(axis=2).astype(np.float32)
    width=max(float(threshold)*.5,1.0)
    return np.clip((difference-float(threshold)+width)/(2*width),0,1)


# Backward-compatible name used by older callers/tests.
def postprocess_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    return cleanup_probability(mask.astype(np.float32)/255.0,config)[1]
