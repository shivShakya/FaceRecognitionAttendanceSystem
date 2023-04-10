import numpy as np 
import matplotlib.pyplot as plt
import pywt 
import cv2
import json
import shutil
import os
from PIL import Image 
import requests
from io import BytesIO
import pandas as pd
from bs4 import BeautifulSoup
import shutil
import os
import pickle 
#Ml libraries
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix




#convert video into set of Imageg
def getImageFromVideo(url,name):
       video = cv2.VideoCapture(url)
      
       if not os.path.exists(f"Img_Collect/{name}"):
                  os.makedirs(f"Img_Collect/{name}")
       frame_count = 0
       while True:
           ret, frame = video.read()
           if not ret:
                 break
           cv2.imwrite(f"Img_Collect/{name}/{name}_{frame_count:05d}.jpg", frame)
           frame_count += 1


def getCroppedImage(url):
       try:
         face_cascade = cv2.CascadeClassifier('backend/haarcascades/haarcascade_frontalface_default.xml')
         eye_cascade = cv2.CascadeClassifier('backend/haarcascades/haarcascade_eye.xml')
        # response = requests.get(url)
        # img = np.array(Image.open(BytesIO(response.content)))
         img = cv2.imread(url)
         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
         faces = face_cascade.detectMultiScale(gray, 1.3 ,5)
         if faces is not None:
              for (x,y,w,h) in faces:
                   roi_gray = gray[y:y+h,x:x+w]
                   roi_color = img[y:y+h,x:x+w]
                   eyes = eye_cascade.detectMultiScale(roi_gray)
              if len(eyes) >=2:
                     return roi_color
     
           
       except:
            return None
       

#load files in cropped folder to use further
def load():
      #load path 
    path_to_data = "./Img_Collect/"
    path_to_cr_data = "./Img_Collect/cropped/"

      # in list save all the folder in dataset folder
    img_dirs = []
    for entry in os.scandir(path_to_data):
        if entry.is_dir():
             img_dirs.append(entry.path)
    print(img_dirs)
      # chicek if file already exist 
    if os.path.exists(path_to_cr_data):
          shutil.rmtree(path_to_cr_data)
          os.mkdir(path_to_cr_data)
      
      # list & dictionary
    cropped_image_dirs = []
    face_name_dict = {}
    class_name = {}
    for img_dir in img_dirs:
        count = 1
        name = img_dir.split('/')[-1]
      
        for entry in os.scandir(img_dir):
            cropped_img = getCroppedImage(entry.path)
            if cropped_img is not None:
                   cropped_folder = path_to_cr_data+ name
                   if not os.path.exists(cropped_folder):
                          os.makedirs(cropped_folder)
                          cropped_image_dirs.append(cropped_folder)
                          print("generated cropped image in folder",cropped_folder)
                   
                   cropped_file_name = name + str(count) + ".png"
                   cropped_file_path = cropped_folder  + "/" + cropped_file_name
                   cv2.imwrite(cropped_file_path,cropped_img)

                   if name not in face_name_dict:
                             face_name_dict[name] = []
                   face_name_dict[name].append(cropped_file_path)
                  
                   count +=1
       
    
    if os.path.exists("backend/face_dict.json"):
        with open("backend/face_dict.json", "w") as f:
                json.dump(face_name_dict, f, indent=4)
           
    with open("backend/face_dict.json","rb") as f:
          face_dict = json.load(f)
    
    return face_dict

#image transformer
def w2d(img, mode = 'haar', level = 1):
       imArray = img
       imArray = cv2.cvtColor(imArray,cv2.COLOR_BGR2GRAY)
       imArray = np.float32(imArray)
       imArray /= 255;
       coeffs = pywt.wavedec2(imArray,mode,level=level)

       coeffs_H = list(coeffs)

       coeffs_H[0] *= 0;
       
       imArray_H = pywt.waverec2(coeffs_H,mode)
       imArray_H *= 255;
       
       imArray_H = np.uint8(imArray_H)

       return imArray_H



# get input X,y for Machine learning model
def getInput(face_dict):
      class_dict = {}
      count = 0 
      X =[]
      y = []
     
      for name in face_dict.keys():
            class_dict[name] = count
            count = count +1 

      print(class_dict)
      if os.path.exists("backend/token.json"):
          with open("backend/token.json","w") as f:
                json.dump(class_dict,f)
                
      for name,training_files in face_dict.items():
            for training_image in training_files:
               img = cv2.imread(training_image)
               if img is None:
                     continue
               scalled_raw_img = cv2.resize(img,(150,150))
               img_har = w2d(img,'db1',5)
               scalled_har_img = cv2.resize(img_har,(150,150))
               combined_img =  np.vstack((scalled_raw_img.reshape(150*150*3,1),scalled_har_img.reshape(150*150,1)))
               X.append(combined_img)
               y.append(class_dict[name])
      X = np.array(X).reshape(len(X),90000).astype(float)
      return X,y    



# fit the model using machine learning algorithms
def ModelFit(X,y):
      store = { 'X_tests':[], 'y_tests':[]} 
      X_train,X_tests,y_train,y_test = train_test_split(X,y,test_size = 0.10,random_state=42)
      pipe = Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='rbf', C =10))])
      pipe.fit(X_train,y_train)
      
      for i in X_tests:
          store['X_tests'].append(i)
      for j in y_test:
           store['y_tests'].append(j) 

      with open('backend/test_values.json','w') as f:
            json.dump(store,f)

      

      with open('backend/model.pkl','wb') as f:
                    pickle.dump(pipe,f)
     
       
      with open('backend/model.pkl','rb') as f:
               model = pickle.load(f)

      return model