from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..dataset.loader import Dataset
from .clahe import apply_clahe
from .crop import bounding_box, crop_resize, crop_transform
from .highlights import suppress_highlights
from .masks import build_background, foreground_mask, postprocess_mask
from .sharpen import sharpen


def _process_rgb(image, cfg):
    if cfg["brightness_normalization"]["enabled"]:
        image=np.clip(image.astype(float)*float(cfg["brightness_normalization"]["target_mean"])/max(image.mean(),1),0,255).astype(np.uint8)
    if cfg["clahe"]["enabled"]: image=apply_clahe(image,cfg["clahe"])
    if cfg["highlight_suppression"]["enabled"]: image=suppress_highlights(image,cfg["highlight_suppression"])
    if cfg["sharpen"]["enabled"]: image=sharpen(image,cfg["sharpen"])
    return image


def preprocess_dataset(dataset: Dataset, config: dict[str,Any], output: Path, quality: list[dict]) -> dict:
    dirs={name:output/"processed"/name for name in ("rgb","masks","rgba","edges","previews")}
    for d in dirs.values(): d.mkdir(parents=True,exist_ok=True)
    mcfg=config["mask"]; scale=float(mcfg["analysis_scale"])
    indices=np.linspace(0,len(dataset.frames)-1,min(int(mcfg["background_samples"]),len(dataset.frames)),dtype=int)
    samples=[]
    for i in indices:
        im=cv2.imread(str(dataset.image_path(dataset.frames[i])))
        samples.append(cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA))
    background=build_background(samples)
    small_masks=[]
    total=len(dataset.frames); interval=max(1,total//10)
    for index,frame in enumerate(dataset.frames,1):
        im=cv2.imread(str(dataset.image_path(frame)))
        small=cv2.resize(im,(background.shape[1],background.shape[0]),interpolation=cv2.INTER_AREA)
        if mcfg["mode"] == "external":
            ext=dataset.root/"masks"/(Path(frame.image).stem+".png"); mask=cv2.imread(str(ext),0)
            if mask is None: raise FileNotFoundError(f"External mask missing: {ext}")
            mask=cv2.resize(mask,(background.shape[1],background.shape[0]),interpolation=cv2.INTER_NEAREST)
            mask=postprocess_mask(mask,mcfg)
        else: mask=foreground_mask(small,background,mcfg)
        small_masks.append(mask)
        if index==1 or index%interval==0 or index==total: logging.info("生成 Mask：%d/%d",index,total)
    union=np.maximum.reduce(small_masks)
    box=bounding_box(union,float(config["crop"]["margin_ratio"])) if config["crop"]["enabled"] else (0,0,union.shape[1],union.shape[0])
    configured_size=config["crop"].get("output_size")
    quality_map={r["image"]:r for r in quality}; transforms={}
    for index,(frame,mask) in enumerate(zip(dataset.frames,small_masks),1):
        image=cv2.imread(str(dataset.image_path(frame)))
        small=cv2.resize(image,(background.shape[1],background.shape[0]),interpolation=cv2.INTER_AREA)
        sx=image.shape[1]/background.shape[1]; sy=image.shape[0]/background.shape[0]
        full_box=(int(round(box[0]*sx)),int(round(box[1]*sy)),int(round(box[2]*sx)),int(round(box[3]*sy)))
        out_size=tuple(configured_size) if configured_size else (full_box[2]-full_box[0],full_box[3]-full_box[1])
        rgb=crop_resize(_process_rgb(image,config["preprocessing"]),full_box,out_size)
        full_mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        out_mask=crop_resize(full_mask,full_box,out_size,nearest=True); out_mask=(out_mask>127).astype(np.uint8)*255
        stem=Path(frame.image).stem
        cv2.imwrite(str(dirs["rgb"]/(stem+".jpg")),rgb,[cv2.IMWRITE_JPEG_QUALITY,95])
        cv2.imwrite(str(dirs["masks"]/(stem+".png")),out_mask)
        rgba=cv2.cvtColor(rgb,cv2.COLOR_BGR2BGRA); rgba[:,:,3]=out_mask; cv2.imwrite(str(dirs["rgba"]/(stem+".png")),rgba)
        edges=cv2.Canny(out_mask,50,150); cv2.imwrite(str(dirs["edges"]/(stem+".png")),edges)
        preview_size=(600,450)
        preview=np.hstack((crop_resize(small,box,preview_size),cv2.cvtColor(cv2.resize(out_mask,preview_size,interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR),cv2.resize(rgb,preview_size,interpolation=cv2.INTER_AREA)))
        cv2.putText(preview,quality_map[frame.image]["status"],(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
        cv2.imwrite(str(dirs["previews"]/(stem+".jpg")),preview,[cv2.IMWRITE_JPEG_QUALITY,85])
        transforms[frame.image]=crop_transform(full_box,out_size)
        if index==1 or index%interval==0 or index==total: logging.info("写入处理结果：%d/%d",index,total)
    metadata={"mask_mode":mcfg["mode"],"common_crop":list(map(int,box)),"analysis_scale":scale,"transforms":transforms}
    (output/"processed"/"crop_transforms.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata
