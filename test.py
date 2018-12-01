
from __future__ import division
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
import skimage.io as io


#
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)
# dataDir='..'
# dataType='val2017'
# coco=COCO('../The_Pose/database/coco/annotations/instances_train2017.json'.format(dataDir,dataType))
# catIds = coco.getCatIds(catNms=['person'])
#
#
# catIds = coco.getCatIds(catNms=['person']);
# imgIds = coco.getImgIds(catIds=catIds );
# img = coco.loadImgs(imgIds[6])[0]
#
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()
#
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
# annFile = '../The_Pose/database/coco/annotations/person_keypoints_train2017.json'.format(dataDir,dataType)
# coco_kps=COCO(annFile)
# plt.imshow(I); plt.axis('off')
# ax = plt.gca()
# annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco_kps.loadAnns(annIds)
# coco_kps.showAnns(anns)
# print 'kk'

a = np.zeros((1,2))
b = np.array ([1,2,3])
print 6%4
print b%3