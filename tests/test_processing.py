import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from openscan_eval.config import load_config
from openscan_eval.dataset.loader import load_dataset
from openscan_eval.preprocessing.crop import bounding_box, crop_transform
from openscan_eval.preprocessing.masks import postprocess_mask
from openscan_eval.preprocessing.pipeline import preprocess_dataset
from openscan_eval.quality.report import analyze_quality
from openscan_eval.quality.sharpness import sharpness_metrics


def dataset_at(path):
    with (path/"positions.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["image","position_index","phi_deg","theta_deg"]); w.writerow(["a.jpg",1,0,0])
    Image.fromarray(np.tile(np.arange(128,dtype=np.uint8),(128,1))).convert("RGB").save(path/"a.jpg")
    return load_dataset(load_config(),path)


def test_sharpness_detects_edges():
    sharp=np.zeros((64,64),np.uint8); sharp[:,::2]=255
    blurred=cv2.GaussianBlur(sharp,(0,0),5)
    assert sharpness_metrics(sharp)[0] > sharpness_metrics(blurred)[0]


def test_manual_quality_override(tmp_path):
    dataset=dataset_at(tmp_path); (tmp_path/"quality_override.csv").write_text("image,status\na.jpg,GOOD\n")
    rows=analyze_quality(dataset,load_config(),tmp_path/"out")
    assert rows[0]["status"]=="GOOD" and rows[0]["reason"]=="manual_override"


def test_mask_postprocessing_removes_small_component():
    mask=np.zeros((100,100),np.uint8); mask[20:80,20:80]=255; mask[2:4,2:4]=255
    out=postprocess_mask(mask,{"opening_kernel":0,"closing_kernel":0,"min_component_ratio":.01,"keep_largest_component":True,"fill_holes":False})
    assert out[30,30]==255 and out[2,2]==0


def test_crop_transform():
    mask=np.zeros((100,200),np.uint8); mask[20:80,50:150]=255
    box=bounding_box(mask,0); transform=crop_transform(box,(100,60))
    assert box==(50,20,150,80) and transform["scale_xy"]==[1.0,1.0]


def test_full_frame_preprocessing_preserves_source_dimensions(tmp_path):
    dataset=dataset_at(tmp_path); config=load_config(); config["crop"]["enabled"]=False; config["crop"]["output_size"]=None
    config["mask"]["analysis_scale"]=0.5
    preprocess_dataset(dataset,config,tmp_path/"out",[{"image":"a.jpg","status":"GOOD"}])
    rgb=cv2.imread(str(tmp_path/"out/processed/rgb/a.jpg")); mask=cv2.imread(str(tmp_path/"out/processed/masks/a.png"),0)
    assert rgb.shape[:2]==(128,128) and mask.shape==(128,128)
