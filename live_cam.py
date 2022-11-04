
import numpy as np
import cv2
import os
from mods.modules import detect,nothing,thresholding,preprocessing,calc_foreground_percentage,pixel_cm
# from modules import detect, nothing,thresholding,preprocessing,calc_foreground_percentage,pixel_cm
import RPi.GPIO as GPIO
import serial 
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(19,GPIO.IN)
ser = serial.Serial("/dev/ttyS0", 115200)


cap =  cv2.VideoCapture(-1)
value = []
status = True
kernel = np.ones((5, 5), np.uint8)
try :
    folder = 'result'
    exist = os.path.exists(folder)
    if not exist:
        os.makedirs(folder)
        print('diretory created')
except FileExistsError:
    print('directory already exist')

def nothing(x):
    pass

def lidar():
    while True:
        #time.sleep(0.1)
        count = ser.in_waiting
        # print(count)
        if count > 8:
            recv = ser.read(9)   
            ser.reset_input_buffer() 
            
            if recv[0] == 0x59 and recv[1] == 0x59:    
                distance = recv[2] + recv[3] * 256
                # strength = recv[4] + recv[5] * 256
                # print(f'distance : {distance}')
                ser.reset_input_buffer()
                return distance
            else:
                print('')
        else:
            pass
def stop():
    GPIO.cleanup()
# if __name__ == '__main__':
def main(path,format_name):
    global status,y,x,h,w,cx,cx,cy,hsv_min,hsv_max

    while True:
        # print('Running')
        # show frame original
        # _, frame = cap.read()
        # copy_frame = frame.copy()
        # print('sucess')
        # cv2.imshow('original',frame)
        # calcuate LIDAR (jarak)
        # jarak = lidar()
        # print(jarak)
        # press button (input jarak)
        # if GPIO.input(19):
        #   print('tombol ditekan)
        try:
            _, frame = cap.read()
            copy_frame = frame.copy()

            if ser.is_open == False:
                ser.open()

            jarak = lidar()
            cv2.putText(copy_frame, f'jarak:{str(jarak)} cm', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),1)
            cv2.imshow('original_cam',copy_frame)

            if (cv2.waitKey(27) & 0xFF == ord('c') or GPIO.input(19)):
                ratio = pixel_cm(jarak) 
                print(f'captured, jarak {jarak}, ratio {ratio}px/cm')
                ser.close()
                for i in range(8):
                    print('====='*i)
                    sleep(0.2)
                print('segmentation stage')
                cv2.namedWindow("HSV Value")
                cv2.createTrackbar("H MIN", "HSV Value", 0, 179, nothing)
                cv2.createTrackbar("S MIN", "HSV Value", 0, 255, nothing)
                cv2.createTrackbar("V MIN", "HSV Value", 0, 255, nothing)
                cv2.createTrackbar("H MAX", "HSV Value", 179, 255, nothing)
                cv2.createTrackbar("S MAX", "HSV Value", 255, 255, nothing)
                cv2.createTrackbar("V MAX", "HSV Value", 255, 255, nothing)
                
                # ratio = 60   
            # save original image 
                cv2.imwrite(f'{path}/{format_name}_original.jpg',frame)
                # cap.release()
                # cv2.destroyAllWindows()
            # detect image from saved image
                path_im = f'{path}/{format_name}_original.jpg'
                # frame = cv2.imread(path_im)
            # do tunning segmentation calcuate luas area
                # process = preprocessing(frame)
                # blur = process.blur(3,1)
                while status:
                    # path_im = 'result/original.jpg'
                    frame = cv2.imread(path_im)
                    process = preprocessing(frame)
                    blur = process.blur(3,1)
                    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                    h_min = cv2.getTrackbarPos("H MIN", "HSV Value")
                    s_min = cv2.getTrackbarPos("S MIN", "HSV Value")
                    v_min = cv2.getTrackbarPos("V MIN", "HSV Value")
                    h_max = cv2.getTrackbarPos("H MAX", "HSV Value")
                    s_max = cv2.getTrackbarPos("S MAX", "HSV Value")
                    v_max = cv2.getTrackbarPos("V MAX", "HSV Value")
                    
                    lower_value = np.array([h_min, s_min, v_min])
                    upper_value = np.array([h_max, s_max, v_max])
                    
                    hsv_min="MIN H:{} S:{} V:{}".format(h_min,s_min,v_min)
                    hsv_max = "MAX H:{} S:{} V:{}".format(h_max, s_max, v_max)
                    
                    mask = cv2.inRange(hsv, lower_value, upper_value)
                    result = cv2.bitwise_and(frame, frame, mask=mask)
                    dilation = cv2.dilate(result, kernel, iterations=2)
                    
                    cx,cy,luas,edge,x,y,w,h = detect(dilation)  
                    cv2.imshow('original_cam',dilation)  
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,45,0),3)                
                    if ((cv2.waitKey(27) & 0xFF == ord('c')) or GPIO.input(19)):
                        status = False

                        # GPIO.cleanup()

                    
                # handle x dan y jika = 0
                if(y>1 and x> 1):
                    crop_img = copy_frame[y:y+h, x:x+w]
                    cv2.imwrite(f'{path}/{format_name}_image_crop.jpg',crop_img)
                    crop_img = cv2.imread(f'{path}/{format_name}_image_crop.jpg')
                    
                    # cv2.imshow("Frame croping", crop_img)
                    th_img,percent = thresholding(crop_img)
                    cv2.imwrite(f'{path}/{format_name}_image_segmentation.jpg',th_img)
                    
                    percent = calc_foreground_percentage(th_img)
                    txt_percent = "percentage area : {} %".format(percent)
                cv2.circle(frame,(cx,cy),5,(255,5,5),-1)
                # cv2.imshow('object',img_detection)
                # luas bounding box w/scale_factor dan H/scale_factor
                luas_area = round((w/ratio)*(h/ratio),2)
                luas_luka = round(((percent)*luas_area),4)
                print(f'1cmPersegi : {luas_area} ')
                # print(f'center point X:{cx}, Y:{cy}, Luas_BBox:{luas}, Luas_boundingBox :{luas_area}cm^2')
                # cv2.putText(frame, hsv_min, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                # cv2.putText(frame, hsv_max, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                # cv2.putText(frame,txt_percent,(10,110),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(frame, f'luas area BBox : {str(luas_area)} cm2', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                cv2.putText(frame, f'luas area luka : {str(luas_luka)} cm2', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                cv2.putText(frame, f'P/L luka : {str(round(w/ratio,2))} cm, {str(round(h/ratio,2))} cm', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)


                cv2.imwrite(f'{path}/{format_name}_bbox.jpg',dilation)
                cv2.imwrite(f'{path}/{format_name}_original.jpg',frame)

            # saved image with calcuate 
                # cv2.imwrite('result/image_crop.jpg',crop_img)
                # cv2.imwrite('result/image_segmentation.jpg',th_img)
            # saved result.txt file
                value.append(f'luas area luka : {luas_area} cm2')
                with open(f'{path}/{format_name}_result.txt', 'w') as result_txt:
                    result_txt.write(str(value))
                    
                    print('end')
                    
                    cv2.destroyAllWindows()
                    break
        except KeyboardInterrupt:   # Ctrl+C
            if ser != None:
                ser.close()
