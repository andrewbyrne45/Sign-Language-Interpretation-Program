# program to capture single image from webcam in python
  
# importing OpenCV library
import cv2 as cv
  
# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that. 
# 0 = webcam
# 1 = external camera
#cam_port = 0
#cam = cv.VideoCapture(cam_port)

# initialize the webcam
cam = cv.VideoCapture(0)
  
# reading the input using the camera
result, image = cam.read()
  
# If image will detected without any error, 
# show result
if result:
  
    # showing result, it take frame name and image 
    # output
    cv.imshow("TestImage", image)
  
    # saving image in local storage
    cv.imwrite("TestImage.png", image)
  
    # if keyboard interrupt occurs, destroy image 
    # window
    cv.waitKey(0)
    cv.destroyWindow("TestImage")
  
# if captured image is corrupted, moving to else part
else:
    print("No image detected. Please try again")