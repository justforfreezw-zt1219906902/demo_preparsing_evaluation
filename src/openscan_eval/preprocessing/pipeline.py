from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..dataset.loader import Dataset
from ..quality.report import analyze_quality
from .clahe import apply_clahe
from .crop import bounding_box, crop_resize, crop_transform, normalized_roi_to_box
from .highlights import suppress_highlights
from .masks import background_model, background_probability, cleanup_probability
from .sharpen import sharpen
from .u2net import U2NetSegmenter


def apply_optional_preprocessing(image: np.ndarray, cfg: dict) -> np.ndarray:
    result=image.copy()
    if cfg["brightness_normalization"]["enabled"]:
        result=np.clip(result.astype(float)*float(cfg["brightness_normalization"]["target_mean"])/max(result.mean(),1),0,255).astype(np.uint8)
    if cfg["clahe"]["enabled"]: result=apply_clahe(result,cfg["clahe"])
    if cfg["highlight_suppression"]["enabled"]: result=suppress_highlights(result,cfg["highlight_suppression"])
    if cfg["sharpen"]["enabled"]: result=sharpen(result,cfg["sharpen"])
    return result


def _checkerboard(height:int,width:int,size:int)->np.ndarray:
    yy,xx=np.indices((height,width)); pattern=((xx//size+yy//size)%2)[...,None]
    return np.where(pattern,205,245).astype(np.uint8).repeat(3,axis=2)


def _preview(original,soft,processed,frame,row,cfg):
    size=tuple(cfg["preview"]["panel_size"]); original=cv2.resize(original,size,interpolation=cv2.INTER_AREA)
    alpha=cv2.resize(soft,size,interpolation=cv2.INTER_LINEAR); processed=cv2.resize(processed,size,interpolation=cv2.INTER_AREA)
    overlay=original.copy(); color=np.zeros_like(overlay); color[:]=(40,180,255); overlay=np.clip(overlay*.65+color*.35*alpha[...,None]+original*.35*(1-alpha[...,None]),0,255).astype(np.uint8)
    checker=_checkerboard(size[1],size[0],int(cfg["preview"]["checker_size"])); composite=np.clip(processed*alpha[...,None]+checker*(1-alpha[...,None]),0,255).astype(np.uint8)
    canvas=np.hstack((original,overlay,composite)); colors={"GOOD":(0,190,0),"WARNING":(0,165,255),"REJECT":(0,0,230)}
    text=f"{frame.image}  phi={frame.phi_deg:g}  theta={frame.theta_deg:g}  sharp={row['sharpness']:.1f}  {row['status']}"
    cv2.rectangle(canvas,(0,0),(canvas.shape[1],42),(20,20,20),-1);cv2.putText(canvas,text,(12,29),cv2.FONT_HERSHEY_SIMPLEX,.7,colors[row["status"]],2,cv2.LINE_AA)
    return canvas


def _segmenter(config):
    mode=config["mask"]["mode"]
    return U2NetSegmenter.from_config(config["mask"]) if mode=="u2net" else None


def _external_probability(dataset,frame,shape,config):
    path=dataset.root/config["mask"].get("external_dir","masks")/(Path(frame.image).stem+".png")
    mask=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if mask is None: raise FileNotFoundError(f"External mask missing: {path}")
    if mask.ndim==3: mask=mask[:,:,3] if mask.shape[2]==4 else cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    return cv2.resize(mask.astype(np.float32)/255.0,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)


def preprocess_dataset(dataset: Dataset, config: dict[str,Any], output: Path, quality=None, write_outputs: bool=True) -> dict:
    """Crop, segment, assess quality, process RGB, and emit RGBA plus previews."""
    processed=output/"processed"; rgba_dir=processed/"rgba"; preview_dir=processed/"previews"
    if write_outputs: rgba_dir.mkdir(parents=True,exist_ok=True);preview_dir.mkdir(parents=True,exist_ok=True)
    debug_dir=processed/"masks" if write_outputs and config.get("debug",{}).get("save_masks",False) else None
    if debug_dir: debug_dir.mkdir(parents=True,exist_ok=True)
    first=cv2.imread(str(dataset.image_path(dataset.frames[0]))); height,width=first.shape[:2]
    crop_cfg=config["crop"]; mode=crop_cfg.get("mode","manual"); mask_cfg=config["mask"]
    if not crop_cfg.get("enabled",True): common_box=(0,0,width,height)
    elif mode=="manual": common_box=normalized_roi_to_box(crop_cfg["roi_xyxy"],width,height)
    elif mode=="auto_from_masks": common_box=None
    else: raise ValueError(f"Unsupported crop mode: {mode}")
    segmenter=_segmenter(config)
    background=None
    if mask_cfg["mode"]=="background_subtraction":
        indices=np.linspace(0,len(dataset.frames)-1,min(int(mask_cfg["background_samples"]),len(dataset.frames)),dtype=int)
        samples=[cv2.imread(str(dataset.image_path(dataset.frames[i]))) for i in indices]
        background=background_model(samples)
    probabilities=[]; crops=[]; total=len(dataset.frames); interval=max(1,total//10)
    # auto_from_masks must segment full frames first; manual mode segments the configured crop directly.
    for index,frame in enumerate(dataset.frames,1):
        image=cv2.imread(str(dataset.image_path(frame))); box=common_box or (0,0,width,height); x0,y0,x1,y1=box; crop=image[y0:y1,x0:x1]
        if mask_cfg["mode"]=="u2net": probability=segmenter.predict(crop)
        elif mask_cfg["mode"]=="external":
            full=_external_probability(dataset,frame,image.shape[:2],config); probability=full[y0:y1,x0:x1]
        elif mask_cfg["mode"]=="background_subtraction": probability=background_probability(crop,background[y0:y1,x0:x1],mask_cfg["difference_threshold"])
        else: raise ValueError(f"Unsupported mask mode: {mask_cfg['mode']}")
        probabilities.append(probability);crops.append(crop)
        if index==1 or index%interval==0 or index==total: logging.info("分割：%d/%d",index,total)
    if common_box is None:
        union=np.maximum.reduce([(p>=mask_cfg["alpha_threshold"]).astype(np.uint8)*255 for p in probabilities])
        local=bounding_box(union,float(crop_cfg.get("margin_ratio",0))); common_box=local
        x0,y0,x1,y1=common_box
        crops=[cv2.imread(str(dataset.image_path(f)))[y0:y1,x0:x1] for f in dataset.frames]
        probabilities=[p[y0:y1,x0:x1] for p in probabilities]
    soft_masks=[];binary_masks=[]
    for p in probabilities:
        soft,binary=cleanup_probability(p,mask_cfg);soft_masks.append(soft);binary_masks.append(binary)
    regions={f.image:(im,mask) for f,im,mask in zip(dataset.frames,crops,binary_masks)}
    quality_rows=analyze_quality(dataset,config,output,regions);quality_by_name={r["image"]:r for r in quality_rows}
    if not write_outputs: return {"metadata":None,"quality":quality_rows}
    resize_cfg=crop_cfg["resize"]; requested=resize_cfg.get("output_size") if resize_cfg.get("enabled",True) else None
    transforms={}
    for index,(frame,image,soft,binary) in enumerate(zip(dataset.frames,crops,soft_masks,binary_masks),1):
        out_size=tuple(requested) if requested else (image.shape[1],image.shape[0])
        rgb=apply_optional_preprocessing(image,config["preprocessing"])
        if (rgb.shape[1],rgb.shape[0])!=out_size: rgb=cv2.resize(rgb,out_size,interpolation=cv2.INTER_AREA)
        if (soft.shape[1],soft.shape[0])!=out_size: soft=cv2.resize(soft,out_size,interpolation=cv2.INTER_LINEAR)
        alpha=np.clip(np.round(soft*255),0,255).astype(np.uint8);rgba=cv2.cvtColor(rgb,cv2.COLOR_BGR2BGRA);rgba[:,:,3]=alpha
        stem=Path(frame.image).stem;cv2.imwrite(str(rgba_dir/(stem+".png")),rgba)
        if debug_dir: cv2.imwrite(str(debug_dir/(stem+".png")),alpha)
        preview=_preview(image,soft,apply_optional_preprocessing(image,config["preprocessing"]),frame,quality_by_name[frame.image],config)
        cv2.imwrite(str(preview_dir/(stem+".jpg")),preview,[cv2.IMWRITE_JPEG_QUALITY,88])
        transforms[frame.image]=crop_transform(common_box,out_size)
        if index==1 or index%interval==0 or index==total: logging.info("写入 RGBA/Preview：%d/%d",index,total)
    metadata={"crop_mode":mode,"crop_box_original_xyxy":list(map(int,common_box)),"resize_enabled":bool(resize_cfg.get("enabled",True)),"transforms":transforms}
    (processed/"crop_transforms.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return {"metadata":metadata,"quality":quality_rows}
