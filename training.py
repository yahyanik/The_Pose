
from __future__ import division
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
import skimage.io as io

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir='..'
dataType='val2017'
coco_kps=COCO('../The_Pose/database/coco/annotations/person_keypoints_train2017.json'.format(dataDir,dataType))
catIds = coco_kps.getCatIds(catNms=['person'])


cats = coco_kps.loadCats(coco_kps.getCatIds())
print cats[0]


nms=[cat['name'] for cat in cats]
imgIds = coco_kps.getImgIds(catIds=coco_kps.getCatIds(catNms=['person']))
for i in range (1,2):
    img = coco_kps.loadImgs(imgIds[i])[0]
    x =  img['width']
    y = img['height']
    annIds = coco_kps.getAnnIds(imgIds=img['id'])
    anns = coco_kps.loadAnns(annIds)[0]
    print anns['keypoints']
    anns['keypoints'][0] /= x
    anns['keypoints'][15] /= x
    anns['keypoints'][18] /= x
    anns['keypoints'][21] /= x
    anns['keypoints'][24] /= x
    anns['keypoints'][1] /= y
    anns['keypoints'][16] /= y
    anns['keypoints'][19] /= y
    anns['keypoints'][22] /= y
    anns['keypoints'][25] /= y
    print anns['keypoints']
    keypoints = np.array(anns['keypoints'])






