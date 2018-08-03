from awsapp import app

from flask import render_template
from flask import Flask, request, Response
from ASLPredictor import ASLPredictor
import cv2
import numpy as np
import json

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/classify',methods=['POST'])
def classify():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    asl = ASLPredictor()
    letter = asl.predictor(img)
    response = json.dumps({'message':letter})
    return response

    

