from flask import Flask, redirect, url_for, render_template, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import cv2
import numpy as np
import base64
import os
from PIL import Image
from model.main import main as model

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
        response = model(image)
        
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)