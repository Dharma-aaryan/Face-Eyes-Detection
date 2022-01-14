import  numpy as np
import cv2

#to turn on the camera and capture live video
cap=cv2.VideoCapture(0)

#Creating a Cascade Classifier Object
#Cascading is basically used for determining face features.
#haar-cascade is basically a set of positive and negative image where,
#the positive images are images we want our classifier to identify and,
#the negative images are images of everything else, which do not want to detect.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#haar cascae for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret,frame = cap.read()

    #converting the image from RGB to Gray.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #here we are using the face_classifier which is an object loaded with haarcascade_frontalface_default.xml,
    #We are using an inbuilt function with it called the detectMultiScale.
    #This function will help us to find the features/locations of the new image.
    #The way it does is, it will use all the features from the face_classifier object to detect the features of the new image.
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    #in this part we are basically identifying the face and eyes and creating a rectange box around.
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)

        roi_gray = gray[y:y+w,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        eyes= eye_cascade.detectMultiScale(roi_gray,1.3,5)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),5)

    cv2.imshow('frame',frame)

    #on pressing the alphabet q, it will terminate the process.
    if(cv2.waitKey(1)==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
