from face_recognizer.algorithms import LBPHAlgorithm, EigenFaceAlgorithm, FisherFaceAlgorithm
from models import Person, Image, Session
from face_recognizer.processors import grayscale, resize, crop
from face_recognizer.detectors import BasicFaceDetector
from face_recognizer.recognizers import FaceRecognizer
from helpers import show_image, load_faces
from timeit import default_timer as timer
import PIL as pil
import numpy as np
import cv2
import os
import sys

print ("\n".join(sys.path))

images_path = 'data/images'
extra_images_path = 'data/extra'
training_data_path = 'data/eigen_faces_data.yml'
haarcascades_data_path = 'data/haarcascade_frontalface_default.xml'
lfw_dataset_path = 'data/lfw'
students_dataset_path = 'data/students'

# Setup face recognition
detector = BasicFaceDetector(classifier_data_path=haarcascades_data_path)
recognizer = FaceRecognizer(
    #    algorithm=FisherFaceAlgorithm())
    #    algorithm=LBPHAlgorithm())
    #    algorithm=EigenFaceAlgorithm(data_path=training_data_path))
    algorithm=EigenFaceAlgorithm())


def train_recognizer(save=False):
    print("Processing images...")
    start = timer()
    size = (100, 100)

    dirty_images, dirty_labels = load_faces()
    images = []
    labels = []
    for idx, img in enumerate(dirty_images):
        face = detector.detect_single(img)
        if not face:
            continue
        images.append(crop(img, face))
        labels.append(dirty_labels[idx])

    images = [grayscale(image) for image in images]
    images = [resize(image, size) for image in images]
    end = timer()
    print("Images Processed: %d" % len(images))
    print("Total processing time: %d" % (end - start))

    print("Training recognizer...")
    start = timer()
    recognizer.add(images, labels)
    recognizer.train()
    if save:
        recognizer.save(training_data_path)
    end = timer()
    print("Total processing time: %d" % (end - start))


# First Example
# print("Should be face: ", labels[0])
# prediction = recognizer.predict(images[0])
# print(prediction)

# DB Example
# session = Session()
# student = session.query(Person).get(7)  # Obama
# student_img = student.images[0]
# student_img_data = cv2.imread(student_img.path)
# student_img_data = grayscale(student_img_data)
# student_img_data = resize(student_img_data, size)
# print("Attempting to find student: %d %s" % (student.id, student.name))
# prediction = recognizer.predict(student_img_data)
# predicted = session.query(Person).get(prediction[0])
# print("Found student: %d %s" % (predicted.id, predicted.name))


# Obama Recognizer
# obama_img = cv2.imread(os.path.join(extra_images_path, 'obama.1.jpg'))
# obama_img = grayscale(obama_img)
# obama_img = resize(obama_img, size)
# obama_id = 2
# prediction = recognizer.predict(obama_img)
# print(prediction)

# Alan Recognizer
# alan_img1 = cv2.imread(os.path.join(extra_images_path, 'alan.1.jpg'))
# alan_img2 = cv2.imread(os.path.join(extra_images_path, 'alan.2.jpg'))
# alan_img3 = cv2.imread(os.path.join(extra_images_path, 'alan.3.jpg'))
# alan_img3 = grayscale(alan_img3)
# alan_img3 = detector.detect(alan_img3)
# alan_img3 = resize(alan_img3, size)
# alan_img2 = grayscale(alan_img2)
# alan_img2 = detector.detect(alan_img2)
# alan_img2 = resize(alan_img2, size)
# images.append(alan_img3)
# labels.append(999)
# prediction = recognizer.predict(alan_img2)
# print(prediction)
