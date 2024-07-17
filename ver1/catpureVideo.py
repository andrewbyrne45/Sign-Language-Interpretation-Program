# program to capture video from webcam in python
  
# importing OpenCV library
import cv2 as cv
  
# initialize the camera
cam = cv.VideoCapture(0)
  
stop = False

# while loop to run webcam until input is deteected
while stop==False:
    # 'ret(return)' will determine whether the steaming was successful
    # 'frame' will capture each frame
    ret, frame = cam.read()

    # if steaming was successful, then show frames in quick succession
    if ret == True:
        # show image on 'test' window
        cv.imshow("test", frame)
        # wait one second for user input
        key = cv.waitKey(1000)
        # ord() converts a character to its unicode value
        if key == ord("f"):
            # stop program
            stop = True

# release wbecam for future programs if necessary
cam.release()
# close all windows
cv.destroyAllWindows()