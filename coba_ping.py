from mods.ping import *
import cv2
from time import sleep
status = True
while status:
    jarak = read_depth(18)
    print(f'jarak : {jarak} cm')
    sleep(0.5)
    if ((cv2.waitKey(27) & 0xFF == ord('c')) ):
        status = False
        GPIO.cleanup()
    