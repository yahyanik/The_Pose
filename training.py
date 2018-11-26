
from pycocotools.coco import COCO
import pandas as pd
import json

import numpy as np

import matplotlib.pyplot as plt
import pylab

dataDir='..'
dataType='val2017'
annFile = '../The_Pose/database/coco/annotations/person_keypoints_val2017.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

with open(annFile) as data_file:
    data = json.load(data_file)

print data


# print labels