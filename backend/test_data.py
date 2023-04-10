
import numpy as np 
import matplotlib.pyplot as plt
import pywt 
import cv2
import json
import shutil
import os


#convert video into set of Imageg
def getImageFromVideo(url):
       video = cv2.VideoCapture(url)
      
       if not os.path.exists('dataset_test/test'):
                  os.makedirs('dataset_test/test')
       frame_count = 0
       while True:
           ret, frame = video.read()
           if not ret:
                 break
           cv2.imwrite(f"dataset_test/test/test_{frame_count:05d}.jpg", frame)
           frame_count += 1
       

# function for wavelet transformation
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

# function for cropping the images 
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
       
#saved path
def load():
    path_to_data = "./dataset_test/"
    path_to_cr_data = "./dataset_test/cropped_test/"
    cropped_image_dirs = []
    face_test_dict = {}

# store image diractory name in list

    img_dirs = []
    for entry in os.scandir(path_to_data):
         if entry.is_dir():
               img_dirs.append(entry.path)
    print(img_dirs)


#create cropped test function
    if os.path.exists(path_to_cr_data):
           shutil.rmtree(path_to_cr_data)
    os.mkdir(path_to_cr_data)


# save all cropped files into new cropped folder

    for img_dir in img_dirs:
      count = 1
      name = img_dir.split('/')[-1]
      print(name)
      
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
                   
                   if name not in face_test_dict:
                             face_test_dict[name] = []
                   face_test_dict[name].append(cropped_file_path)
                  
                   count +=1
                  


#write a json file with all the names of images
    if  os.path.exists("test_dict.json"):
      with open("test_dict.json","w") as f:
             json.dump(face_test_dict,f)
             print('successfull')
     
    return face_test_dict


def getTestInput(test_url):
    
    with open(test_url,'rb') as f:
           test_dict =  json.load(f)
           print(test_dict)
    X_test = []
    # read that file and apply operations
    for name,testing_files in  test_dict.items():
       for testing_image in testing_files:
               img = cv2.imread(testing_image)
               if img is None:
                     continue
               scalled_raw_img = cv2.resize(img,(150,150))
               img_har = w2d(img,'db1',5)
               scalled_har_img = cv2.resize(img_har,(150,150))
               combined_img =  np.vstack((scalled_raw_img.reshape(150*150*3,1),scalled_har_img.reshape(150*150,1)))
               X_test.append(combined_img)
               

    #transform X_test              
    X_test = np.array(X_test).reshape(len(X_test),90000).astype(float)
    return X_test