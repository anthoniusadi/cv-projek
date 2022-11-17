# test tombol capture
from time import sleep
import cv2
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(19,GPIO.IN)
# GPIO.output(19) = 
print(GPIO.input(19))

try:
    while True:
        # sleep(1)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            GPIO.cleanup()
            print('clean')
            break
        if GPIO.input(19):
            print('tombol ditekan')
            # sleep(2)
        else:
            print('nothing')
    # sleep(10)
finally:
    print('cleanup')
    GPIO.cleanup()
