from __future__ import annotations

import cv2
import numpy as np


def normalized_roi_to_box(roi, width: int, height: int) -> tuple[int,int,int,int]:
    if len(roi)!=4: raise ValueError("crop.roi_xyxy must contain four normalized values")
    x0,y0,x1,y1=map(float,roi)
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1):
        raise ValueError("crop.roi_xyxy must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return int(round(x0*width)),int(round(y0*height)),int(round(x1*width)),int(round(y1*height))


def resolve_manual_crop(config: dict, width: int, height: int) -> tuple[int,int,int,int]:
    if not config.get("enabled",True): return (0,0,width,height)
    if config.get("mode","manual")!="manual": raise ValueError("resolve_manual_crop requires crop.mode=manual")
    return normalized_roi_to_box(config["roi_xyxy"],width,height)


def bounding_box(mask: np.ndarray, margin_ratio: float = 0.0) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if not len(xs): return (0, 0, mask.shape[1], mask.shape[0])
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    margin = int(max(x1-x0, y1-y0) * margin_ratio)
    return max(0,x0-margin), max(0,y0-margin), min(mask.shape[1],x1+margin), min(mask.shape[0],y1+margin)


def crop_resize(image: np.ndarray, box: tuple[int,int,int,int], size: tuple[int,int], nearest=False) -> np.ndarray:
    x0,y0,x1,y1 = box
    return cv2.resize(image[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def crop_transform(box: tuple[int,int,int,int], size: tuple[int,int]) -> dict:
    x0,y0,x1,y1 = map(int,box); w,h=map(int,size)
    return {"source_box_xyxy": [x0,y0,x1,y1], "output_size_wh": [w,h],
            "scale_xy": [float(w/(x1-x0)), float(h/(y1-y0))], "offset_xy": [-x0, -y0]}
