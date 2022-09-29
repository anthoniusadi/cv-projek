
from unicodedata import name
import cv2
import numpy as np
from modules import detect,thresholding,preprocessing,calc_foreground_percentage,pixel_cm
cap =  cv2.VideoCapture(-1)

kernel = np.ones((5, 5), np.uint8)

if __name__ == '__main__':
    # show frame original
    ret, frame = cap.read()
    copy_frame = frame.copy()
    cv2.imshow('original',frame)
    # calcuate LIDAR (jarak)
    jarak = lidar()
    print(jarak)
    # press button (input jarak)
    if cv2.waitKey(27) & 0xFF == ord('c'):
        ratio = pixel_cm(jarak)    
    # save original image 
        cv2.imwrite('result/original.jpg',frame)
        cap.release()
        cv2.destroyAllWindows()
    # detect image from saved image
        path_im = 'resut/original.jpg'
        frame = cv2.imread('image_original',path_im)
    # do tunning segmentation calcuate luas area
        cx,cy,luas,edge,x,y,w,h = detect(frame)    
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,45,0),3)
        # handle x dan y jika = 0
        if(y>1 and x> 1):
            crop_img = copy_frame[y:y+h, x:x+w]
            cv2.imshow("Frame croping", crop_img)
            th_img,percent = thresholding(crop_img)
            # percent = calc_foreground_percentage(th_img)
            
        txt_percent = "percentage area : {} %".format(percent)
        cv2.circle(frame,(cx,cy),5,(255,5,5),-1)
        # cv2.imshow('object',img_detection)
        # perhitungan pixwl_cm
        ratio = pixel_cm(jarak_kamera)
        # luas bounding box w/scale_factor dan H/scale_factor
        luas_area = round((w/ratio)*(h/ratio),2)
        luas_luka = round(((percent/100)*luas_area),2)
        print(f'1cmPersegi : {luas_area} ')
        # print(f'center point X:{cx}, Y:{cy}, Luas_BBox:{luas}, Luas_boundingBox :{luas_area}cm^2')
        cv2.putText(frame, hsv_min, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, hsv_max, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame,txt_percent,(10,110),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'luas area BBox : {str(luas_area)} cm2', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'luas area luka : {str(luas_luka)} cm2', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (37, 55, 195), 2)

    # saved image with calcuate 
        cv2.imwrite('result/image_crop',crop_img)
        cv2.imwrite('result/image_segmentation',th_img)
    # saved result.txt file
    