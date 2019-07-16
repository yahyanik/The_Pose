
from __future__ import division
from pycocotools.coco import COCO
import numpy as np
import cv2


class data (object):

    def DataReshape(self):
        coco_kps, imgIds = self.DataRead('/media/yahya/9EEA399CEA39721F/Users/yahya/Desktop/__DATASET__/database/coco/annotations/person_keypoints_train2017.json', 'train2017')
        # training = self.labeling(coco_kps, imgIds, 'train')
        dataset_training = (coco_kps, imgIds)


        coco_kps, imgIds = self.DataRead('/media/yahya/9EEA399CEA39721F/Users/yahya/Desktop/__DATASET__/database/coco/annotations/person_keypoints_val2017.json', 'val2017')
        # val = self.labeling(coco_kps, imgIds, 'val')
        dataset_val = (coco_kps, imgIds)


        return (dataset_training, dataset_val)


    def DataRead(self,addr, type):
        dataDir='..'
        dataType=type
        coco_kps=COCO(addr.format(dataDir, dataType))
        imgIds = coco_kps.getImgIds(catIds=coco_kps.getCatIds(catNms=['person']))   #get image IDs that have human in them
        return (coco_kps, imgIds)

    def labeling (self, coco_kps, imgIds, filename = 'thrain_val'):

        new_labels = {}         #palceholder for all labels
        # for i in range (0,len(imgIds)):
        for i in range (0,len(imgIds)):
        # for i in range (7,10):           # go through each image

            img = coco_kps.loadImgs(imgIds[i])[0]   #get the image to have its dimenssion and get its annotation
            # imgFile = cv2.imread('../The_Pose/database/coco/images/train2017/'+img['file_name'])

            x = img['width']
            y = img['height']
            annIds = coco_kps.getAnnIds(imgIds=imgIds[i], iscrowd=None)
    # print annIds

            label = np.zeros((10,10,26), dtype=np.float16)    #placeholder for the labels in the annotations

            for id in annIds:
                anns = coco_kps.loadAnns(id)

            # print anns
                key = anns[0]['keypoints']
                bbox = anns[0]['bbox']
                center = [bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2]

                # row, column, channels = imgFile.shape
                # for iii in xrange(row): # to make a mask for the image and cahnge every pixel to black
                #     for jjj in xrange(column):
                #         imgFile[iii,jjj] = [255,255,255]
                #
                # # print key
                # cv2.circle(imgFile, (key[0],key[1]), 2, (0, 0, 255), -1)
                # cv2.circle(imgFile, (key[15],key[16]), 2, (0, 0, 255), -1)
                # cv2.circle(imgFile, (key[18],key[19]), 2, (0, 0, 255), -1)
                # cv2.circle(imgFile, (key[21],key[22]), 2, (0, 0, 255), -1)
                # cv2.circle(imgFile, (key[24],key[25]), 2, (0, 0, 255), -1)
                # cv2.circle(imgFile, (key[27], key[28]), 2, (0, 0, 255), -1)
                # cv2.circle(imgFile, (key[30], key[31]), 2, (0, 0, 255), -1)
                # cv2.circle(imgFile, (int(center[0]), int(center[1])),4,(255,0,200),-1)
                # cv2.rectangle(imgFile, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,255,0),1)
                #
                # cv2.line(imgFile, (int(x/10), 0), (int(x/10), y), (255, 0, 0), 1)
                # cv2.line(imgFile, (int(2*x / 10), 0), (int(2*x / 10), y), (255, 0, 0), 1)
                # cv2.line(imgFile, (int(3 * x / 10), 0), (int(3 * x / 10), y), (255, 0, 0), 1)
                # cv2.line(imgFile, (int(4 * x / 10), 0), (int(4 * x / 10), y), (255, 0, 0), 1)
                # cv2.line(imgFile, (int(5 * x / 10), 0), (int(5 * x / 10), y), (255, 0, 0), 1)
                # cv2.line(imgFile, (int(6 * x / 10), 0), (int(6 * x / 10), y), (255, 0, 0), 1)
                # cv2.line(imgFile, (int(7 * x / 10), 0), (int(7 * x / 10), y), (255, 0, 0), 1)
                # cv2.line(imgFile, (int(8 * x / 10), 0), (int(8 * x / 10), y), (255, 0, 0), 1)
                # cv2.line(imgFile, (int(9 * x / 10), 0), (int(9 * x / 10), y), (255, 0, 0), 1)
                #
                # cv2.line(imgFile, (0, int(y/10)), (x, int(y/10)), (255, 0, 0), 1)
                # cv2.line(imgFile, (0, int(2*y / 10)), (x, int(2*y / 10)), (255, 0, 0), 1)
                # cv2.line(imgFile, (0, int(3*y / 10)), (x, int(3*y / 10)), (255, 0, 0), 1)
                # cv2.line(imgFile, (0, int(4*y / 10)), (x, int(4*y / 10)), (255, 0, 0), 1)
                # cv2.line(imgFile, (0, int(5*y / 10)), (x, int(5*y / 10)), (255, 0, 0), 1)
                # cv2.line(imgFile, (0, int(6*y / 10)), (x, int(6*y / 10)), (255, 0, 0), 1)
                # cv2.line(imgFile, (0, int(7*y / 10)), (x, int(7*y / 10)), (255, 0, 0), 1)
                # cv2.line(imgFile, (0, int(8*y / 10)), (x, int(8*y / 10)), (255, 0, 0), 1)
                # cv2.line(imgFile, (0, int(9*y / 10)), (x, int(9*y / 10)), (255, 0, 0), 1)



                key[0] /= x         #putting the labels in the new 10*10 window dimention
                key[15] /= x
                key[18] /= x
                key[21] /= x
                key[24] /= x
                key[27] /= x
                key[30] /= x
                center[0] /= x
                key[1] /= y
                key[16] /= y
                key[19] /= y
                key[22] /= y
                key[25] /= y
                key[28] /= y
                key[31] /= y
                center[1] /= y

                x1 = bbox[2]/(x/10)
                x2 = bbox[3]/(y/10)


            # print anns['keypoints']
                keypoints = np.array([key[0],key[1],key[15],key[16],key[18],key[19],key[21],
                                      key[22],key[24],key[25],key[27],key[28],key[30],key[31],center[0], center[1]])

                keypoints *= 10             #information to put images in which cell
                id = keypoints.astype(int)
                keypoints_new = keypoints - id
        # print keypoints
        # print keypoints_new

                if key[2] != 0:                         #putting data in their respected palce and change label to show what part is detected
                    label[int (id[0]), int (id[1]), 0] = keypoints_new[0]
                    label[int (id[0]), int (id[1]), 1] = keypoints_new[1]
                    label[int (id[0]), int (id[1]), 2] = key[2]/2       #5 or 10 for nose visible 2
                    # label[int(id[0]), int(id[1]), 2] = key[2]
                if key[17] != 0:
                    label[int(id[2]), int(id[3]), 3] = keypoints_new[2]
                    label[int(id[2]), int(id[3]), 4] = keypoints_new[3]
                    label[int (id[2]), int (id[3]), 5] = key[17]/2    #6 or 11 for left sholder
                if key[20] != 0:
                    label[int(id[4]), int(id[5]), 6] = keypoints_new[4]
                    label[int(id[4]), int(id[5]), 7] = keypoints_new[5]
                    label[int (id[4]), int (id[5]), 8] = key[20]/2    #7 or 12 for right sholder
                if key[23] != 0:
                    label[int(id[6]), int(id[7]), 9] = keypoints_new[6]
                    label[int(id[6]), int(id[7]), 10] = keypoints_new[7]
                    label[int (id[6]), int (id[7]), 11] = key[23]/2   #8 or 13 for left arm
                if key[26] != 0 :
                    label[int(id[8]), int(id[9]), 12] = keypoints_new[8]
                    label[int(id[8]), int(id[9]), 13] = keypoints_new[9]
                    label[int (id[8]), int (id[9]), 14] = key[26]/2   #9 or 14 for right arm
                if key[29] != 0:
                    label[int(id[10]), int(id[11]), 15] = keypoints_new[10]
                    label[int(id[10]), int(id[11]), 16] = keypoints_new[11]
                    label[int (id[10]), int (id[11]), 17] = key[29]/2   #9 or 14 for left hand
                if key[32] != 0:
                    label[int(id[12]), int(id[13]), 18] = keypoints_new[12]
                    label[int(id[12]), int(id[13]), 19] = keypoints_new[13]
                    label[int (id[12]), int (id[13]), 20] = key[32]/2   #9 or 14 for right hand

                label[int(id[14]), int(id[15]), 21] = 1 # pc =1
                label[int(id[14]), int(id[15]), 22] = keypoints_new[14]  # center
                label[int(id[14]), int(id[15]), 23] = keypoints_new[15]
                label[int(id[14]), int(id[15]), 24] = x1              # bounding size
                label[int(id[14]), int(id[15]), 25] = x2



                # print 'this is the outcome', label[4, 5, :]

            # cv2.imshow('image1', imgFile)
            # cv2.waitKey(0)
            new_labels[imgIds[i]] = label

            if (i+1)%10000 == 0:
                print ('10000 images are recorded')


        # f = open("new_labels_"+filename+".txt","w")
        print ('data is saved successfully')
        # f.write(str(new_labels))
        # f.close()
        print ('this many images here' , len (new_labels))
        return new_labels

#########################################################################################################
    # def ImagSet (self, coco_kps, imgIds):
    #
    #     for i in range (0,len(self.imgIds)):
    #         img = coco_kps.loadImgs(self.imgIds[i])[0]






if __name__ == '__main__':

   data = data()
   dataset_training, dataset_val = data.DataReshape()













