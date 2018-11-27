from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
import skimage.io as io

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir='..'
dataType='val2017'
coco_kps=COCO('../The_Pose/database/coco/annotations/person_keypoints_train2017.json'.format(dataDir,dataType))
imgIds = coco_kps.getImgIds(catIds=coco_kps.getCatIds(catNms=['person']))

for i in range (1,2):
    img = coco_kps.loadImgs(imgIds[i])[0]
    annIds = coco_kps.getAnnIds(imgIds=img['id'])
    anns = coco_kps.loadAnns(annIds)
    print anns