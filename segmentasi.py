import cv2
import numpy as np

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


def thresholding(images,x,y):
    img = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv_img = (255-thresh1)
    img_thresh = cv2.imshow('Binary Threshold', thresh1)
   
    pixel_black = cv2.countNonZero(thresh1)
    pixel_white = cv2.countNonZero(inv_img)

    print("Number of dark pixels:")
    print(pixel_black)
    print("Number of white pixels:")
    print(pixel_white)
    # h, w = images.shape()
    luas_total = x*y
    percentage = pixel_black / luas_total
    print(f"Percentage of foreground:{percentage*100},value foreground : {percentage}")
    return img_thresh,percentage


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
        print(f'nilai w {round(l,1)}, milai p {round(p,1)},angle {angle}')
        # end
        x,y,w,h = cv2.boundingRect(c)
        luas = (y-(y+h))*(x-(x+w))
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # temp.append(luas)
        # print(temp)
        # cv2.imshow('roi',roi)
    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        if(luas > 18000):
            # print(f'koordinat : {x,y}, luasan : {w,h}, luas : {luas}' )
            M = cv2.moments(c)
            if M['m00'] != 0:
                # cx1= int(M['m01'])
                # cx2= int(M['m00'])
                # cx3= int(M['m10'])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                print(cx,cy)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3) 
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
    cv2.createTrackbar("H MAX", "HSV Value", 179, 179, nothing)
    cv2.createTrackbar("S MAX", "HSV Value", 255, 255, nothing)
    cv2.createTrackbar("V MAX", "HSV Value", 255, 255, nothing)

    while True:
        global temp
        temp = []
        percent = 0


        # _, frame = cap.read()
        # coba data 1 dan data 4
        frame = cv2.imread('data/penggaris2.jpg')
        copy_frame = frame.copy()
        blur = cv2.GaussianBlur(frame,(9,9),5)
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
        dilation = cv2.dilate(result, kernel, iterations=4)
        cx,cy ,luas,canny,x,y,w,h= detect(dilation)
        # handle x dan y jika = 0
        if(y>1 and x> 1):
            crop_img = copy_frame[y:y+h, x:x+w]
            cv2.imshow("Frame croping", crop_img)
            th_img,percent = thresholding(crop_img,cx,cy)
            # percent = calc_foreground_percentage(th_img)
            

        txt_percent = "percentage area : {}".format(percent)

        cv2.circle(frame,(cx,cy),5,(255,5,5),-1)
        # cv2.imshow('object',img_detection)
        print(f'keterangan X:{cx},Y:{cy},Luas_foreground:{luas},Luas_boundingBox :{cx*cy}')
        cv2.putText(frame, hsv_min, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, hsv_max, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame,txt_percent,(10,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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