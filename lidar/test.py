import serial 
from time import sleep
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(19,GPIO.IN)
# ser = serial.Serial("/dev/serial0", 115200)
ser = serial.Serial("/dev/ttyS0", 115200)
# GPIO.input(19) = 
print(GPIO.input(19))
def getTFminiData():
    while True:
        # sleep(0.4)
        #time.sleep(0.1)
        count = ser.in_waiting
        # print(count)
        if count > 8:
            if GPIO.input(19):
                print('tombol ditekan')
                # cv2.imwrite('data/cek.jpg',frame)
                GPIO.cleanup()
                break
            recv = ser.read(9)   
            ser.reset_input_buffer() 
            # type(recv), 'str' in python2(recv[0] = 'Y'), 'bytes' in python3(recv[0] = 89)
            # type(recv[0]), 'str' in python2, 'int' in python3 
            sleep(0.3)
            if recv[0] == 0x59 and recv[1] == 0x59:     #python3
                distance = recv[2] + recv[3] * 256
                # strength = recv[4] + recv[5] * 256
                print(f'distance : {distance}')
                ser.reset_input_buffer()
            else:
                print('')
        else:
            pass


if __name__ == '__main__':
    print('run!')
    try:
        if ser.is_open == False:
            ser.open()

            
        getTFminiData()
    except KeyboardInterrupt:   # Ctrl+C
        if ser != None:
            ser.close()