import os
import cv2
import numpy as np
from models import Person, Session, Image
from faker import Faker
import PIL as pil


def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img2arr(img):
    pilimg = pil.Image.open(img)
    pilimg.show()
    return np.array(pilimg)


def load_faces():
    images = []
    labels = []
    session = Session()
    students = session.query(Person)
    for student in students:
        for image in student.images:
            img = cv2.imread(image.path)
            images.append(img)
            labels.append(student.id)
    return (images, labels)


def import_unlabeled_dataset(path):
    fake = Faker()
    session = Session()
    people = dict()
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        uid = int(filename.split('.')[1])
        if uid not in people:
            people[uid] = Person(name=fake.name())
            session.add(people[uid])
            session.flush()
        image = Image(path=filepath, person_id=people[uid].id)
        session.add(image)
    session.commit()


def clear_db():
    session = Session()
    session.query(Person).delete()
    session.query(Image).delete()
    session.commit()


def import_lfw_dataset(path, clear=True):
    if clear:
        clear_db()
    with open(os.path.join(path, 'lfw-names.txt')) as f:
        person_list = f.readlines()
        person_list = [s.strip() for s in person_list]

    session = Session()
    count = 0
    for person in person_list:
        name, img_count = person.split('\t')
        img_count = int(img_count)
        if img_count < 10:
            continue
        if count > 30:
            break
        print("%s %d" % (name, img_count))
        person = Person(name=name)
        session.add(person)
        session.flush()
        count = count + 1
        image_folder = os.path.join(path, name)
        img_count = 0
        for imgpath in os.listdir(image_folder):
            if img_count > 12:
                break
            imgpath = os.path.join(image_folder, imgpath)
            image = cv2.imread(imgpath)
            if not isinstance(image, np.ndarray):
                continue
            img_count = img_count + 1
            image = Image(path=imgpath, person_id=person.id)
            session.add(image)
        session.flush()
    session.commit()


def import_students_dataset(path):
    fake = Faker()
    session = Session()
    people = dict()
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        uid = int(filename.split('_')[1].split('.')[0])
        if uid not in people:
            people[uid] = Person(name=fake.name())
            session.add(people[uid])
            session.flush()
        image = Image(path=filepath, person_id=people[uid].id)
        print("%s: %d %d" % (people[uid].name, uid, people[uid].id))
        tmp = cv2.imread(filepath)
        show_image(tmp)
        session.add(image)
    session.commit()
