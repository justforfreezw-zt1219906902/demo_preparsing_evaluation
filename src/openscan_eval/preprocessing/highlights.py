import cv2
import numpy as np

def suppress_highlights(image, config):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); mask=gray>int(config["threshold"])
    blurred=cv2.GaussianBlur(image,(0,0),5); result=image.astype(float)
    result[mask]=(1-float(config["strength"]))*result[mask]+float(config["strength"])*blurred[mask]
    return np.clip(result,0,255).astype(np.uint8)
