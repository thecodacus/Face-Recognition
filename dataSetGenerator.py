#Copyright Anirban Kar (anirbankar21@gmail.com)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier(path+r'\Classifiers\face.xml')
i=0
offset=50
name=raw_input('enter your id')
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.waitKey(100)
    if i>20:
        cam.release()
        cv2.destroyAllWindows()
        break

