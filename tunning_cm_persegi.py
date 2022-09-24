from operator import inv
import cv2
import numpy as np

class preprocessing:
    def __init__(self,img):
        self.img= img
    def blur(self,kernel_size,iterasi):
        return cv2.GaussianBlur(self.img,(kernel_size,kernel_size),iterasi)
        # return cv2.GaussianBlur(self.img,(kernel,kernel),iterasi)
    def show(self):
        return cv2.imshow("original",self.img)
    def dilate(self,kernel_size,iterasi):
        return cv2.dilate(self.img, kernel_size, iterations=iterasi)
        

def calc_foreground_percentage(img):
    pixel_black = cv2.countNonZero(img)

    print("Number of dark pixels:")
    print(pixel_black)

    h, w = img.shape
    luas_total = h*w
    percentage = pixel_black / luas_total
    print(f"Percentage of foreground:{percentage*100},value : {percentage}")
    # print(pixel_black / luas_total * 100)    
    
    return percentage


def thresholding(images):
    kernel_th = np.ones((3,3),np.uint8)
    
    img = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # inv_img = (255-thresh1)
    thresh1 = cv2.erode(thresh1, kernel_th, iterations=1)
    
    size = np.size(img)
    img_thresh = cv2.imshow('Binary Threshold', thresh1)
    # img_inv = cv2.imshow('Binary invers', inv_img)

    #  0 = hitam 
    background = cv2.countNonZero(thresh1)
    foreground = size - background
    # pixel_black = cv2.countNonZero(thresh1)
    

    print(f"Number of foreground pixels: {foreground}, background pixels: {background}")


    # h, w = images.shape()
    # luas_total = x*y
    luas = foreground+background
    percentage = round((foreground / luas)*100,2)
    # percentage = round(percentage*100,2)
    print(f"Percentage of foreground in pixel:{percentage}%")
    return img_thresh,percentage

def pixel_cm(jarak):
    persamaan = ((0.09104895*(np.power(jarak,2))) + (-6.2577418*jarak) + 123.02010343130996)
    print(persamaan)
    return persamaan
    
def detect(frame):
    # global x,y,w,h
    # x,y,w,h = 0,0,0,0
    cx,cy =0 , 0
    luas=0
    edge = cv2.Canny(frame,30,100,3)
    contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 23 agstus
        rect = cv2.minAreaRect(c)
        (a,b),(l,p),angle=rect
        # box =cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.circle(frame,int(a),int(b),5,(0,0,255),-1)
        # end
        x,y,w,h = cv2.boundingRect(c)
        # luas = (y-(y+h))*(x-(x+w))
        luas= w*h
   
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # temp.append(luas)
        # print(temp)
        # cv2.imshow('roi',roi)
    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        if(luas > 180):
            # print(f'koordinat : {x,y}, luasan : {w,h}, luas : {luas}' )
            M = cv2.moments(c)
            if M['m00'] != 0:
                # cx1= int(M['m01'])
                # cx2= int(M['m00'])
                # cx3= int(M['m10'])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
          
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 
                return cx,cy,luas,edge,x,y,w,h
                
            else:
                cx,cy =0 , 0
        else:
            cx,cy=0,0
      
    # print(f'X:{cx},Y:{cy}')
    # print(f'X:{cx1},Y:{cx2},{cx3}')
    
    luas,edge,x,y,w,h=0,0,0,0,0,0 # fungsinya supaya ketika tidak ada kontuk nilainya dikembalikan 0 semua
    return cx,cy,luas,edge,x,y,w,h
def nothing(x):
    pass
kernel = np.ones((5, 5), np.uint8)

if __name__ == '__main__':

    # cap = cv2.VideoCapture(0)

    cv2.namedWindow("HSV Value")
    cv2.createTrackbar("H MIN", "HSV Value", 0, 179, nothing)
    cv2.createTrackbar("S MIN", "HSV Value", 0, 255, nothing)
    cv2.createTrackbar("V MIN", "HSV Value", 0, 255, nothing)
    cv2.createTrackbar("H MAX", "HSV Value", 179, 255, nothing)
    cv2.createTrackbar("S MAX", "HSV Value", 255, 255, nothing)
    cv2.createTrackbar("V MAX", "HSV Value", 255, 255, nothing)

    while True:
        global temp
        temp = []
        percent = 0
        jarak_kamera = 25

        # _, frame = cap.read()
        # coba data 1 dan data 4
        frame = cv2.imread('data/distance25cm.jpg')
        # frame = cv2.resize(frame, (480, 320))
        copy_frame = frame.copy()
        process = preprocessing(frame)
        im = process.show()
        # blur = cv2.GaussianBlur(frame,(7,7),5)
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
        # dilation = process.dilate(kernel,2)
        dilation = cv2.dilate(result, kernel, iterations=2)
        cx,cy ,luas,canny,x,y,w,h= detect(dilation)
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
        
        

        cv2.imshow("HSV Value", frame)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Frame Mask", result)
        # cv2.imshow("Frame blur", canny)
        cv2.imshow("Frame dilation", dilation)
        


        key = cv2.waitKey(1)
        if key == 27:
            # cv2.imwrite('generate_croped.jpg',crop_img)
            # cv2.imwrite('generate_th.jpg',th_img)

            break

    cv2.destroyAllWindows()