

from pycocotools.coco import COCO
import numpy as np

import matplotlib.pyplot as plt
import pylab

dataDir='..'
dataType='val2017'
annFile = '../The_Pose/database/coco/annotations/person_keypoints_train2017.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

print coco_kps

annIds = coco_kps.getAnnIds()



# print labels