# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:53:35 2018

@author: Shriya Prasad
"""

from flask import Flask, request, abort, jsonify
from models import Person, Image, Session
from helpers import img2arr
from face_recognizer import FaceRecognizer, EigenFaceAlgorithm, BasicFaceDetector, \
    grayscale, resize, crop
import cv2
import PIL as pil

# Constants
size = (100, 100)

# Data Paths
training_data_path = 'data/eigen_faces_data.yml'
haarcascades_data_path = 'data/haarcascade_frontalface_default.xml'

app = Flask(__name__)
session = Session()
recognizer = FaceRecognizer(
    algorithm=EigenFaceAlgorithm(data_path=training_data_path))
detector = BasicFaceDetector(
    classifier_data_path=haarcascades_data_path, scale_factor=2)


def find_person(id):
    return session.query(Person).get(id)


@app.route("/v1/predict", methods=['POST'])
def recognize_unprocessed():
    img = request.files['img']
    if not img:
        abort(400, {'message': 'Img file missing'})
    imgarr = img2arr(img)
    facedim = detector.detect_single(imgarr)
    if not facedim:
        abort(400, {'message': 'Face not detected in img'})
    face = crop(imgarr, facedim)
    face = resize(grayscale(face), size)
    result = recognizer.predict(face)
    if not result:
        abort(404, {'message': 'Person not recognized'})
    person = find_person(result[0])
    if not person:
        abort(404, {'message': 'Person not recognized'})
    return jsonify({'id': person.id, 'name': person.name, 'confidence': result[1]})


@app.route("/v2/predict", methods=['POST'])
def recognize_processed():
    img = request.files['img']
    if not img:
        abort(400, {'message': 'Img file missing'})
    imgarr = img2arr(img)
    result = recognizer.predict(imgarr)
    if not result:
        abort(404, {'message': 'Person not recognized'})
    person = find_person(result[0])
    if not person:
        abort(404, {'message': 'Person not recognized'})
    return jsonify({'id': person.id, 'name': person.name, 'confidence': result[1]})


if __name__ == '__main__':
    app.run(debug=True)
