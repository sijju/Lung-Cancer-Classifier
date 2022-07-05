from __future__ import division, print_function
from pyexpat import model
import sys
import os
import glob
import re
import numpy as np
import h5py
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH1= './ct_incep_best_model.hdf5'
MODEL_PATH2 = './ct_resnet_best_model.hdf5'
MODEL_PATH3 = './ct_vgg_best_model.hdf5'
MODEL_PATH4 = './ct_alex_best_model.hdf5'
model1 = load_model(MODEL_PATH1)
model2 = load_model(MODEL_PATH2)
model3 = load_model(MODEL_PATH3)
model4 = load_model(MODEL_PATH4)

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(path, _model):
    classes_dir = ["Adenocarcinoma","Large cell carcinoma","Normal","Squamous cell carcinoma"]
    img = image.load_img(path, target_size=(350,350))
    norm_img = image.img_to_array(img)/255
    input_arr_img = np.array([norm_img])
    pred = np.argmax(_model.predict(input_arr_img))
    
    return classes_dir[pred]


@app.route('/', methods=['GET'])
def index():
   
    return render_template('index.html',
    data=[{'name':'InceptionV3','accuracy':'83.81%'}, {'name':'Resnet50','accuracy':'84.76%'},{'name':'AlexNet','accuracy':'77.43%'},{'name':'VGG16','accuracy':'79.36%'}]
    )


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
       
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath,'Dataset/Data/test',secure_filename(f.filename))
        f.save(file_path)
        model = request.form.get('comp_select')
        print(model)
       
        if(model=="InceptionV3"):
         preds = model_predict(file_path, model1)
         return preds
        elif(model=="Resnet50"):
         preds = model_predict(file_path,model2)
         return preds
        elif(model=="VGG16"):
         preds = model_predict(file_path,model3)
         return preds
        elif(model=="AlexNet"):
         preds = model_predict(file_path,model4)
         return preds
        
    return None


if __name__ == '__main__':
    app.run(debug=True)