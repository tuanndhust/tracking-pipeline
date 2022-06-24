import tqdm
from charlp_detection.detect import CHARLP
from LPN_LM.yolov5landmark import Yolov5Landmark
import glob
import time
import os
import cv2
import shutil

class LicensePlate:
    def __init__(self):
        self.lpn_lm = Yolov5Landmark()
        self.charlp = CHARLP()
    
    def detect(self,image):
        boxes,img_aligneds=self.lpn_lm.detect(image)
        for box,img_aligned in zip(boxes,img_aligneds):
            box_detects,text = self.charlp.detect(img_aligned)
            cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(255,0,0,),1,1)
            cv2.putText(image,text,(box[0],box[1]),cv2.FONT_HERSHEY_SIMPLEX,(image.shape[2])/7,(0,255,255),1)
            cv2.imshow("img_aligned",img_aligned)
        # cv2.waitKey(0)
        return image

    def predict_single(self,image):
        text=""
        boxes,img_aligneds=self.lpn_lm.detect(image)
        if(len(img_aligneds)>0):
            box_detects,text = self.charlp.detect(img_aligneds[0])
        return text



import tqdm
if __name__ == "__main__":
    X = LicensePlate()
    for path in glob.glob("data_LP_test/*/*"):
        img=cv2.imread(path)
        image=X.detect(img)
        # cv2.imwrite(os.path.join("results",path.split("/")[-1]),image)
        cv2.imshow("image",image)
        cv2.waitKey(0)

