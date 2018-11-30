
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
# print cats[0]


nms=[cat['name'] for cat in cats]
imgIds = coco_kps.getImgIds(catIds=coco_kps.getCatIds(catNms=['person']))

for i in range (6,7):
    img = coco_kps.loadImgs(imgIds[i])[0]
    x =  img['width']
    I = io.imread(img['coco_url'])
    y = img['height']
    annIds = coco_kps.getAnnIds(imgIds=img['id'])
    # print annIds
    label = np.zeros(10,10,15)
    for id in annIds:
        anns = coco_kps.loadAnns(id)
        # print anns
        key = anns[0]['keypoints']

        key[0] /= x
        key[15] /= x
        key[18] /= x
        key[21] /= x
        key[24] /= x
        key[1] /= y
        key[16] /= y
        key[19] /= y
        key[22] /= y
        key[25] /= y
    # print anns['keypoints']
    # keypoints = np.array([key[0],key[1],key[15],anns['keypoints'][16],anns['keypoints'][18],anns['keypoints'][19],anns['keypoints'][21],anns['keypoints'][22],anns['keypoints'][24],anns['keypoints'][25]])














