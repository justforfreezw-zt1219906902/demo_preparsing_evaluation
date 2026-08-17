import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from openscan_eval.config import load_config
from openscan_eval.dataset.loader import load_dataset
from openscan_eval.export.pytorch3d_dataset import export_pytorch3d
from openscan_eval.preprocessing.crop import bounding_box, crop_transform
from openscan_eval.preprocessing.masks import cleanup_probability, postprocess_mask
from openscan_eval.preprocessing.pipeline import apply_optional_preprocessing, preprocess_dataset
from openscan_eval.quality.report import analyze_quality, object_quality_region
from openscan_eval.quality.sharpness import sharpness_metrics


def dataset_at(path, image=None):
    image=np.full((100,200,3),120,np.uint8) if image is None else image
    image[:,100:]=(220,220,220)
    with (path/"positions.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["image","position_index","phi_deg","theta_deg"]);w.writerow(["a.jpg",1,0,0])
    cv2.imwrite(str(path/"a.jpg"),image);masks=path/"masks";masks.mkdir();cv2.imwrite(str(masks/"a.png"),np.full(image.shape[:2],255,np.uint8))
    return load_dataset(load_config(),path)


def external_config():
    config=load_config();config["mask"]["mode"]="external";config["crop"]["resize"]["enabled"]=False
    return config


def test_sharpness_detects_edges():
    sharp=np.zeros((64,64),np.uint8);sharp[:,::2]=255;blurred=cv2.GaussianBlur(sharp,(0,0),5)
    assert sharpness_metrics(sharp)[0]>sharpness_metrics(blurred)[0]


def test_manual_quality_override(tmp_path):
    dataset=dataset_at(tmp_path);(tmp_path/"quality_override.csv").write_text("image,status\na.jpg,GOOD\n")
    rows=analyze_quality(dataset,load_config(),tmp_path/"out")
    assert rows[0]["status"]=="GOOD" and rows[0]["reason"]=="manual_override"


def test_mask_postprocessing_removes_small_component():
    mask=np.zeros((100,100),np.uint8);mask[20:80,20:80]=255;mask[2:4,2:4]=255
    out=postprocess_mask(mask,{"alpha_threshold":.5,"opening_kernel":0,"closing_kernel":0,"min_component_ratio":.01,"keep_largest_component":True,"feather_radius":0})
    assert out[30,30]==255 and out[2,2]==0


def test_soft_alpha_generation():
    probability=np.zeros((30,30),np.float32);probability[8:22,8:22]=.8
    soft,_=cleanup_probability(probability,{"alpha_threshold":.5,"opening_kernel":0,"closing_kernel":0,"min_component_ratio":0,"keep_largest_component":False,"feather_radius":1.5})
    assert np.any((soft>0)&(soft<.8))


def test_crop_transform():
    mask=np.zeros((100,200),np.uint8);mask[20:80,50:150]=255;box=bounding_box(mask,0);transform=crop_transform(box,(100,60))
    assert box==(50,20,150,80) and transform["scale_xy"]==[1.0,1.0]


def test_manual_crop_visibly_changes_rgba_and_preview(tmp_path):
    dataset=dataset_at(tmp_path);config=external_config();config["crop"]["roi_xyxy"]=[.5,0,1,1]
    preprocess_dataset(dataset,config,tmp_path/"out")
    rgba=cv2.imread(str(tmp_path/"out/processed/rgba/a.png"),cv2.IMREAD_UNCHANGED)
    preview=cv2.imread(str(tmp_path/"out/processed/previews/a.jpg"))
    assert rgba.shape[:2]==(100,100) and rgba[:,:,:3].mean()>190
    assert preview is not None and preview.shape[1]==1800


def test_only_rgba_and_previews_produced_by_default(tmp_path):
    dataset=dataset_at(tmp_path);preprocess_dataset(dataset,external_config(),tmp_path/"out")
    processed=tmp_path/"out/processed"
    assert (processed/"rgba").is_dir() and (processed/"previews").is_dir()
    assert not (processed/"rgb").exists() and not (processed/"masks").exists() and not (processed/"edges").exists()


def test_pytorch3d_handoff_contains_only_rgba(tmp_path):
    dataset=dataset_at(tmp_path);result=preprocess_dataset(dataset,external_config(),tmp_path/"out")
    target=export_pytorch3d(dataset,tmp_path/"out",result["quality"])
    assert (target/"rgba/a.png").is_file()
    assert not (target/"images").exists() and not (target/"masks").exists() and not (target/"edges").exists()


def test_optional_preprocessing_switches_change_result():
    image=np.full((64,64,3),100,np.uint8);image[20:40,20:40]=220;cfg=load_config()["preprocessing"]
    assert np.array_equal(apply_optional_preprocessing(image,cfg),image)
    cfg["clahe"]["enabled"]=True
    assert not np.array_equal(apply_optional_preprocessing(image,cfg),image)


def test_object_aware_quality_roi_excludes_background():
    image=np.zeros((100,100,3),np.uint8);image[:,::2]=255;image[35:65,35:65]=100
    mask=np.zeros((100,100),np.uint8);mask[35:65,35:65]=255
    full,_=object_quality_region(image,None);obj,obj_mask=object_quality_region(image,mask)
    assert sharpness_metrics(full)[0]>sharpness_metrics(obj,obj_mask)[0]
