from mods.calibrate import calibrasi
from mods.detect import deteksi
from mods.preparation import prep
from mods.calculate import calc
import cv2
import numpy as np

file_path=''
if __name__ == '__main__':
    # set kalibrasi pixel to cm
    a = calibrasi()
    # load image
    frame = cv2.imread(file_path)
    # preprocessing stage
    b = prep()
    # detect boundary
    c = deteksi()
    # calculate luas 
    d = calc()
    # show to display
