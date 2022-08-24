from cmath import rect
from inspect import Parameter
import cv2
from mesureobj.object_detector import *
import numpy as np


parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
detector = HomogeneousBgDetector()


img = cv2.imread('data/phone_aruco_marker.jpg')

corners,_,_ =cv2.aruco.detectMarkers(img,aruco_dict,parameters=parameters)
print(corners)
aruco_perimeter = cv2.arcLength(corners[0],True)
# pixel to ratio
pixel_cm = aruco_perimeter /20
# print(pixel_cm)

contur =detector.detect_objects(img)

for cnt in contur:
    cv2.polylines(img,[cnt],True,(20,255,5),2)
    rect = cv2.minAreaRect(cnt)
    (x,y),(w,h),angle=rect
    # print(x,y,w,h)
    # print(angle)
    

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print(box)
    cv2.circle(img,(int(x),int(y)),5,(0,255,40),-1)
    cv2.polylines(img,[box],True,(255,0,0),2)
    cv2.putText(img,"lebar {} panjang {}".format(round(w/pixel_cm,2),(round(h/pixel_cm,2))),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(10,20,243),1)
cv2.imshow('image original',img)
cv2.waitKey(0)