
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

    label = np.zeros((10,10,15))

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
        keypoints = np.array([key[0],key[1],key[15],key[16],key[18],key[19],key[21],key[22],key[24],key[25]])

        keypoints *= 10
        id = keypoints.astype(int)
        keypoints_new = keypoints - id
        print keypoints
        print keypoints_new

        if key[2] != 0:
            label[int (id[0]), int (id[1]), 0] = keypoints_new[0]
            label[int (id[0]), int (id[1]), 1] = keypoints_new[1]
            label[int (id[0]), int (id[1]), 2] = key[2]
        if key[17] != 0:
            label[int(id[2]), int(id[3]), 3] = keypoints_new[2]
            label[int(id[2]), int(id[3]), 4] = keypoints_new[3]
            label[int (id[2]), int (id[3]), 5] = key[17]
        if key[20] != 0:
            label[int(id[4]), int(id[5]), 6] = keypoints_new[4]
            label[int(id[4]), int(id[5]), 7] = keypoints_new[5]
            label[int (id[4]), int (id[5]), 8] = key[20]
        if key[23] != 0:
            label[int(id[6]), int(id[7]), 9] = keypoints_new[6]
            label[int(id[6]), int(id[7]), 10] = keypoints_new[7]
            label[int (id[6]), int (id[7]), 11] = key[23]
        if key[26] != 0:
            label[int(id[8]), int(id[9]), 12] = keypoints_new[8]
            label[int(id[8]), int(id[9]), 13] = keypoints_new[9]
            label[int (id[8]), int (id[9]), 14] = key[26]


        # label[id[0], id[1]] = [kepoints_new[2], kepoints_new[3], key[17]]
        # label[id[2], id[3]] = [kepoints_new[2],kepoints_new[3],key[17]]
        # label[id[4], id[5]] = [kepoints_new[4],kepoints_new[5],key[20]]
        # label[id[6], id[7]] = [kepoints_new[6],kepoints_new[7],key[23]]
        # label[id[8], id[9]] = [kepoints_new[8],kepoints_new[9],key[26]]


    print label.shape
    print np.max(label)
    print label[6,3,0]












