from flask import Flask, redirect, url_for, render_template, request, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import cv2
import numpy as np
import pandas as pd
import base64
import os
import io
from PIL import Image
from model.main import main as model
from model.segmentation.multiotsu_segmentation import __main__ as multiotsu_segmentation
from model.segmentation.bitwise_operation import __main__ as bitwise_operation
from model.extraction.main import __main__ as feature_extraction
from model.classification.main import __main__ as classify

from model.preprocessing.rgb_to_gray import __main__ as rgb_to_gray

UPLOAD_FOLDER = 'static/image_uploads'

app = Flask(__name__)
app.secret_key = "RahasiaTau"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

# db = SQLAlchemy(app)

# class Records(db.Model):
#     id = db.column(db.string(100), primary_key=True)s 

@app.route('/')
def home():
    filename = session.get('upload_img', None)
    return render_template("home.html", filename=filename)

@app.route('/api/', methods=['POST'])
def submit_file():
    if 'file' not in request.files:        
        return redirect('/')
    
    file = request.files['file']
    
    if file:
        image = Image.open(file)
        image = np.array(image)
        
        gray_image = rgb_to_gray(image)
        
        mask_image = multiotsu_segmentation(gray_image)        
        
        segmented_image = bitwise_operation(image, mask_image)
        
        features = feature_extraction(segmented_image)
        
        result = classify(features)        
        
        return jsonify({
            "features": features.to_dict(orient='list'),
            "result": "Abnormal" if result else "Normal"
        })
    
@app.route('/api/preprocessing/', methods=['POST'])
def preprocessing():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    
    if file:
        image = Image.open(file)
        image = np.array(image)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = Image.fromarray(gray_image)
        
        img_io = io.BytesIO()
        gray_image.save(img_io, 'JPEG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    
@app.route('/api/segmentation/', methods=['POST'])
def segmentation():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    
    if file:
        image = Image.open(file)
        image = np.array(image)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        mask_image = multiotsu_segmentation(gray_image)
        
        segmented_image = bitwise_operation(image, mask_image)
        
        img_io = io.BytesIO()
        segmented_image = Image.fromarray(segmented_image)
        segmented_image.save(img_io, 'JPEG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    
@app.route('/api/classification/', methods=['POST'])
def classification():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    
    if file:
        image = Image.open(file)
        image = np.array(image)
        
        features = feature_extraction(image)
        
        result = classify(features)
        
        return jsonify({
            'features': features.to_dict(orient='list'),
            'result': "Abnormal" if result else "Normal"
        })

if __name__ == '__main__':
    app.run(debug=True)