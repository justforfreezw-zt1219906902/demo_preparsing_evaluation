import cv2

def apply_clahe(image, config):
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(float(config["clip_limit"]),tuple(config["grid_size"])).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
