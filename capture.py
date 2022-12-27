# test capture image dengan GPIO
import cv2
import RPi.GPIO as GPIO
import datetime as dt
from mods.ping import *
import gc
gc.collect()
GPIO.setmode(GPIO.BCM)
GPIO.setup(19,GPIO.IN)

cap =  cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    jarak = read_depth(18)
    cv2.putText(frame, f'jarak:{str(jarak)} cm', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),1)

    cv2.imshow('frame', frame)
    if GPIO.input(19):
        print('tombol ditekan')
        waktu = str(dt.datetime.now())
        cv2.imwrite(f'data/{waktu}.jpg',frame)
        GPIO.cleanup()
        del frame
        gc.collect()
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print('tombol ditekan')
        waktu = str(dt.datetime.now())
        cv2.imwrite(f'data/{waktu}.jpg',frame)
        GPIO.cleanup()
        break
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()