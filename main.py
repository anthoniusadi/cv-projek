
from email.mime import image
import re
import cv2 
import numpy as np
# import matplotlib as plt

folder = 'data/'
thresh =  20
font = cv2.FONT_HERSHEY_SIMPLEX
def contour_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    ##! nilai 50 dan 400 atur sesuai dengan pencahayaan dan benda yang mau di deteksi
    edged = cv2.Canny(gray, 50, 200) 
    contours, hierarchy = cv2.findContours(edged,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    jumlah = str(len(contours))
    print("Number of Contours found = " + jumlah) 
    jumlah=int(jumlah)
   # print(jumlah)
    ##?perhitungan kontur untuk setiap benda yang terhitung
    for i in range (jumlah):
    ##! perhatikan jika nilai perimeter =0 maka akan muncul error  zero division
        area = cv2.contourArea(contours[i])          
        perimeter = cv2.arcLength(contours[i],True)
        TR=(4*np.pi*area)/(perimeter**2)
####? nilai TR ukur sendiri sesuai hasil dari benda dan labeli nama dengan bentuknya masing masing
        if TR<=0.73:
            bentuk='persegi panjang'
        elif TR>0.73:
            bentuk='persegi '
        else:
            bentuk='lainnya'
        print('Nilai Thinnes Ratio : ', TR)
        print('Bentuk benda :',bentuk)

  #  cv2.imshow('Canny Edges After Contouring', edged)   
    result=cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
    cv2.imshow('result', result) 
    cv2.imshow('Contours', edged) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    # interrupt = cv2.waitKey(10)
    # if interrupt & 0xFF == 27: # esc key
    #     break
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 
def show_im(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.putText(image,f'shape : {image.shape}',(20,40),font,fontScale=1,color=(255,0,0),thickness=2)
    cv2.imshow('original',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_single_im():
    pass

def segmentasi(image_path):
    image_original = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)
    _ ,image =  cv2.threshold(image_gray, 120, 255, cv2.THRESH_OTSU)
    cv2.putText(image,f'shape : {image.shape}',(20,40),font,fontScale=1,color=(255,0,0),thickness=2)
    cv2.imshow('original',image_original)
    cv2.imshow('image_gray',image_gray)

    cv2.imshow('thresh_image',image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def estimate():
    pass
def max_rgb_filter(image):
	# split the image into its BGR components
    (B, G, R) = cv2.split(image)
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0
	# merge the channels back together and return the image
    return cv2.merge([B, G, R])
def color_segmentation(image_path):
    image_original = cv2.imread(image_path)
    hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    lower_color = np.array([30,150,50])
    upper_color = np.array([255,255,180])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(image_original,image_original, mask=mask)
    
    cv2.imshow('original',image_original)

    cv2.imshow('image_hsv',res)

    cv2.imshow('hsv',hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segment_fish(image):
    ''' Attempts to segment the clownfish out of the provided image '''

    # Convert the image into HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Set the orange range
    light_orange = (1, 190, 200)
    dark_orange = (18, 255, 255)

    # Apply the orange mask 
    mask = cv2.inRange(hsv_image, light_orange, dark_orange)

    # Set a white range
    light_white = (0, 0, 0)
    dark_white = (166, 60, 255)

    # Apply the white mask
    mask_white = cv2.inRange(hsv_image, light_white, dark_white)

    # Combine the two masks
    final_mask =  mask_white
    result = cv2.bitwise_and(image, image, mask=final_mask)

    # Clean up the segmentation using a blur
    blur = cv2.GaussianBlur(result, (7, 7), 0)
    return blur

if __name__ == '__main__':
    print('launch')
    # color_segmentation(folder+'data3.jpg')
    im = cv2.imread(folder+'data1.jpg')
    hasil = segment_fish(im)

    # image = cv2.imread(folder+'data3.jpg')

    # filtered = max_rgb_filter(image)
    # recon_image = np.hstack([image, filtered])

    cv2.imshow('recon_image',hasil)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
