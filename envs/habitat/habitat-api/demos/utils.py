import cv2
import pdb
import math
import numpy as np

def proc_rgb(rgb):
    return cv2.resize(np.flip(rgb, axis=2), (300, 300))
