# test capture image dengan GPIO
import cv2
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(19,GPIO.IN)

cap =  cv2.VideoCapture(-1)

while True:

    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if GPIO.input(19):
        print('tombol ditekan')
        cv2.imwrite('data/cek.jpg',frame)
        GPIO.cleanup()
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print('capturing frame')
        cv2.imwrite('data/cek.jpg',frame)
        break
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()