# test tombol capture
from time import sleep
import cv2
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(19,GPIO.IN)
try:
    while True:
        if cv2.waitKey(27) & 0xFF == ord('q'):
            GPIO.cleanup()
            print('clean')
            break
        if GPIO.input(19):
            print('tombol ditekan')
        else:
            print('nothing')
    sleep(0.1)
finally:
    print('cleanup')
    GPIO.cleanup()
