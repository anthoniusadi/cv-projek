
from unicodedata import name
import cv2
import numpy as np
from modules import detect,thresholding,preprocessing,calc_foreground_percentage,pixel_cm
cap =  cv2.VideoCapture(-1)


kernel = np.ones((5, 5), np.uint8)

if __name__ == '__main__':
    # show frame original
    
    # press button
    
    # calcuate LIDAR (jarak)
    
    # save original image 
    
    # detect image from saved image
    
    # do tunning segmentation
    
    # calculate area
    
    # saved image with calcuate 
    
while True:
    global temp
    temp = []
    percent = 0

    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.namedWindow("HSV Value")
    cv2.createTrackbar("H MIN", "HSV Value", 0, 179, nothing)
    cv2.createTrackbar("S MIN", "HSV Value", 0, 255, nothing)
    cv2.createTrackbar("V MIN", "HSV Value", 0, 255, nothing)
    cv2.createTrackbar("H MAX", "HSV Value", 179, 255, nothing)
    cv2.createTrackbar("S MAX", "HSV Value", 255, 255, nothing)
    cv2.createTrackbar("V MAX", "HSV Value", 255, 255, nothing)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print('capturing frame')
        cv2.imwrite('data/original.jpg',frame)
        while True:
        
            copy_frame = frame.copy()
            process = segmentasi.preprocessing(frame)  
            blur = process.blur(3,1)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            
            h_min = cv2.getTrackbarPos("H MIN", "HSV Value")
            s_min = cv2.getTrackbarPos("S MIN", "HSV Value")
            v_min = cv2.getTrackbarPos("V MIN", "HSV Value")
            h_max = cv2.getTrackbarPos("H MAX", "HSV Value")
            s_max = cv2.getTrackbarPos("S MAX", "HSV Value")
            v_max = cv2.getTrackbarPos("V MAX", "HSV Value")

            lower_blue = np.array([h_min, s_min, v_min])
            upper_blue = np.array([h_max, s_max, v_max])
            
            hsv_min="MIN H:{} S:{} V:{}".format(h_min,s_min,v_min)
            hsv_max = "MAX H:{} S:{} V:{}".format(h_max, s_max, v_max)

            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            result = cv2.bitwise_and(frame, frame, mask=mask)
            dilation = cv2.dilate(result, kernel, iterations=2)
            cx,cy ,luas,canny,x,y,w,h= detect(dilation)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,45,0),3)
            if(y>1 and x> 1):
                crop_img = copy_frame[y:y+h, x:x+w]
                cv2.imshow("Frame croping", crop_img)
                th_img,percent = thresholding(crop_img)
            txt_percent = "percentage area : {} %".format(percent)
            cv2.circle(frame,(cx,cy),5,(255,5,5),-1)
            luas_area = round((w/60)*(h/60),2)
            print(f'center point X:{cx}, Y:{cy}, Luas_BBox:{luas}, Luas_boundingBox :{luas_area}cm^2')
            cv2.putText(frame, hsv_min, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(frame, hsv_max, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(frame,txt_percent,(10,110),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'luas area : {str(luas_area)} cm2', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("HSV Value", frame)
            # cv2.imshow("Frame dilation", dilation)

            break
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()