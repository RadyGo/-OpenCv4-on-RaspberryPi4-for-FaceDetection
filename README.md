# OpenCv4-on-RaspberryPi4-for-FaceDetection
Learn how you can use the open-source library OpenCV with a Raspberry Pi to create face and object detection!

<img src="https://github.com/noorkhokhar99/-OpenCv4-on-RaspberryPi4-for-FaceDetection/blob/master/0.jpg">


What is OpenCV?

OpenCv is an open source computer vision and machine learning software library. OpenCv is released under a BSD license making it free for both academic and commercial use. It has C++, Python, and Java interfaces and supports Windows, Linux, Mac OS, iOS, and Android. 

OpenCv was designed for computational efficiency and a strong focus on real-time applications.

How to Install OpenCV on a Raspberry Pi?
To install OpenCV for Python, we need to have Python installed. Since Raspberry Pi's are preloaded with Python, we can install OpenCV directly.

First we need to make sure that Raspberry pi is up to date. Type the following command to update your Raspberry Pi.


<img src="https://github.com/noorkhokhar99/-OpenCv4-on-RaspberryPi4-for-FaceDetection/blob/master/0.png">


Then type the following command in the terminal to get the dependencies required for installing OpenCV on your Raspberry Pi.


<img src="https://github.com/noorkhokhar99/-OpenCv4-on-RaspberryPi4-for-FaceDetection/blob/master/0%20(1).png">



sudo apt-get install libhdf5-dev -y && sudo apt-get install libhdf5-serial-dev -y && sudo apt-get install libatlas-base-dev -y && sudo apt-get install libjasper-dev -y && sudo apt-get install libqtgui4 -y && sudo apt-get install libqt4-test -y

Now we can install OpenCV on Raspberry Pi. Type the following command to install OpenCV 4 for Python on your Raspberry Pi 4, pip/pip3 tells us that OpenCV will get installed for Python.

For python2 installation cmd pip install opencv-contrib-python==4.1.0.25

<img src="https://github.com/noorkhokhar99/-OpenCv4-on-RaspberryPi4-for-FaceDetection/blob/master/0%20(2).png">


For python3 pip3 install opencv-contrib-python==4.1.0.25


<img src="https://github.com/noorkhokhar99/-OpenCv4-on-RaspberryPi4-for-FaceDetection/blob/master/0%20(3).png">




After those steps, OpenCV should be installed. Let's test our work!

Testing OpenCV 

To check whether OpenCV is correctly installed or not, try importing OpenCV by typing:

python

then:

import cv2

<img src="https://github.com/noorkhokhar99/-OpenCv4-on-RaspberryPi4-for-FaceDetection/blob/master/0%20(4).png">


Same cmd line for checking on python3


<img src="https://github.com/noorkhokhar99/-OpenCv4-on-RaspberryPi4-for-FaceDetection/blob/master/0%20(5).png">




If no errors are shown, your installation was successful!

Face Detection Using OpenCV
Let's start by writing the code that will detect face in the Video frame it receives. For this, you need a cascade file.

Type following command to get the cascade file.

wget https://raw.githubusercontent.com/shantnu/Webcam-Face-Detect/master/haarcascade_frontalface_default.xml

wget https://raw.githubusercontent.com/vnosc/master/master/Shared/XML/haarcascade_frontaleye_default


Now run the following code and it will perform face detection on the Video Frame.

import numpy as np
import cv2


# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades


#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()


After running the code, it will draw rectangles around the face and eyes as shown in the picture below.








<img src="https://github.com/noorkhokhar99/-OpenCv4-on-RaspberryPi4-for-FaceDetection/blob/master/0%20(6).png">

