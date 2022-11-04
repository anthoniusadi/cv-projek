from mods.ping import *
import cv2

status = True
while status:
    jarak = read_depth(18)
    print(f'jarak : {jarak} cm' )
    if ((cv2.waitKey(27) & 0xFF == ord('c')) ):
        status = False
        GPIO.cleanup()
    