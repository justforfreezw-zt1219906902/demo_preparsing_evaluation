import cv2
import numpy as np

def sharpen(image, config):
    blur=cv2.GaussianBlur(image,(0,0),float(config["radius"])); amount=float(config["amount"])
    return np.clip(image.astype(float)*(1+amount)-blur.astype(float)*amount,0,255).astype(np.uint8)
