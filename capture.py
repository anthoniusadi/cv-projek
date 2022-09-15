import cv2

cap =  cv2.VideoCapture(2)

while True:

    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print('capturing frame')
        cv2.imwrite('data/distance34cm.jpg',frame)
        break
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()