import argparse
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import apply_instances
import mrcnn.model as modellib
from mrcnn.model import log

import rocksmodel

def load_image(path):
        image = Image.open(path)
        rgbimage = image.convert('RGB')
        w, h = rgbimage.size
        return np.array(rgbimage.getdata()).reshape(h, w, 3).astype(np.uint8)

def detect_and_save(model, path):
    image = load_image(path)
    results = model.detect([image])
    r = results[0]

    apply_instances(image, r['rois'], r['masks'], r['class_ids'],
                    ['back', 'rock'], path, r['scores'])


parser = argparse.ArgumentParser(description="Detect rocks on images")
parser.add_argument("-i", "--image", help="Input image")
parser.add_argument("-f", "--folder", help="Input folder")

args = parser.parse_args()
filename = args.image
folder = args.folder

model = rocksmodel.load_model()

if folder:
    for fname in os.listdir(folder):
        path = os.path.abspath(f"{folder}/{fname}")
        detect_and_save(model, path)
elif filename:
    detect_and_save(model, filename)
