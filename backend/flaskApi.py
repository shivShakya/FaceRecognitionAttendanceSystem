import io
import test_data
import faceRecognition
 
from flask import Flask,request,jsonify
import os
import shutil
import tempfile
import base64
from flask_cors import CORS
import pickle
import json

from moviepy.video.io.VideoFileClip import VideoFileClip

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file provided'
    blob = request.files['video']
    name = request.form.get('name')
    print(name)
    


    # create a directory named video
    
    byte_data = blob.read()
    encoded_data = base64.b64encode(byte_data)
    byte_data = base64.b64decode(encoded_data)
    with io.BytesIO(byte_data) as f:
                   with open(name+'.mp4', 'wb') as out_file:
                               shutil.copyfileobj(f, out_file)


    faceRecognition.getImageFromVideo(name+".mp4",name)
    return jsonify({'message': 'Video converted successfully'})




#prediction 

@app.route('/predict',methods=['POST'])
def predict():
    if 'video' not in request.files:
        return 'No video file provided'
    blob = request.files['video']
    byte_data = blob.read()
    encoded_data = base64.b64encode(byte_data)
    byte_data = base64.b64decode(encoded_data)
    with io.BytesIO(byte_data) as f:
       with open('test.mp4', 'wb') as out_file:
                shutil.copyfileobj(f, out_file)

    test_data.getImageFromVideo("test.mp4")
    face_dict =  test_data.load()
   
    url = "backend/test_dict.json"
    X_test = test_data.getTestInput(url)
    
    #manually
    #with open('model.pkl','rb') as f:
              #model = pickle.load(f)
    
    face_dict =  faceRecognition.load()
    X,y = faceRecognition.getInput(face_dict)

    model = faceRecognition.ModelFit(X,y)

    prediction = model.predict(X_test)
    print(prediction)

    
    
    with open('backend/token.json','rb') as f:
            token = json.load(f)
    # conversion 
    c = [0,0]
    for j in token.values():
          for i in range(len(prediction)):
                if prediction[i] == j:
                                   c[j] += 1
                
    max_value = max(c)
    index = c.index(max_value)
    name = []
    for key,value in token.items():
                if index == value:
                      name.append(key)
   
    
    return jsonify({'message': name[0]})



if __name__ == "__main__":
           app.run()
