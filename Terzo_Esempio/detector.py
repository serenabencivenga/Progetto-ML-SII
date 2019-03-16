import cv2
from PIL import Image
import pickle
import numpy as np
#cam = cv2.VideoCapture(0);
#faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

#rec= cv2.face.LBPHFaceRecognizer_create()
#rec.read('recognizer\\trainningData.yml')
#id=0
#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,1,1,0,0)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainner.yml')


cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(Id==1):
                Id="Serena_Bencivenga"
            elif(Id==2):
                Id="Sam"
        else:
            Id="Sconosciuto"
        cv2.putText(im,str(Id),(x,y+h),fontFace, fontScale, fontColor);

        #cv2.putText(str(Id), (x,y+h),font,fontScale,255 )
    cv2.imshow("Face",im);
    if cv2.waitKey(10) & 0xFF==ord('q'):
         break;
        
cam.release()
cv2.destroyAllWindows()
