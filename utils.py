import base64
from datetime import datetime
import os
import pickle
import sqlite3
import cv2
import keras
import numpy as np
import cnn
face_detection = cv2.CascadeClassifier(
    "resources/haarcascade_frontalface_default.xml")
face_recognition = keras.saving.load_model(os.environ["CNN_PATH"])

con = sqlite3.connect(os.environ["DATABASE_PATH"], check_same_thread=False)
cur = con.cursor()


def reload_result_map():
    with open("ResultsMap.pkl", 'rb') as r:
        cnn.ResultMap = pickle.load(r)


def get_last_user_id():
    cur.execute("select ifnull(max(user_id), 0) from users")
    row = cur.fetchone()
    return row[0]


def get_last_img_id():
    cur.execute("select ifnull(max(dataset_id), 0) from dataset")
    row = cur.fetchone()
    return row[0]


def generate_dataset(user_id):
    face_classifier = cv2.CascadeClassifier(
        "resources/haarcascade_frontalface_default.xml")

    def face_cropped(img, padding=24):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y-padding:y + h +
                               padding, x - padding:x + w + padding]
        return cropped_face

    cap = cv2.VideoCapture(0)

    lastid = get_last_img_id()

    img_id = lastid
    max_imgid = img_id + 40
    count_img = 0

    while True:
        _, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (240, 240))  # type: ignore

            dir = f"dataset/{user_id}"
            if not os.path.exists(dir):
                os.mkdir(dir)
            file_name_path = f"{dir}/{str(img_id)}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', face)
            img_b64 = base64.b64encode(buffer)
            cur.execute(
                'INSERT INTO dataset(user_id, image) VALUES ({}, "{}")'.format(user_id, img_b64))
            con.commit()
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                cap.release()
                cv2.destroyAllWindows()
                break


def detect_face():
    capture = cv2.VideoCapture(0)

    while True:
        _, img = capture.read()

        faces = face_detection.detectMultiScale(img, 1.1, 4)

        for (x, y, w, h) in faces:
            test_img = keras.preprocessing.image.img_to_array(
                img[y:y+64, x:x+64])
            test_img = np.expand_dims(test_img, axis=0)

            coffidance_result = face_recognition.predict(test_img, verbose=0)
            idx = np.argmax(coffidance_result[0])

            cur.execute(f"SELECT name FROM users WHERE user_id = {idx}")
            rows = cur.fetchone()
            face_name = rows[0]

            # Valid face
            if coffidance_result[0][idx] > 0.8:
                cv2.putText(img, face_name, (x, y + h),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                handle_session(idx)
            else:
                cv2.putText(img, "UNKNOWN", (x, y + h),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


scanning_session = {}


def handle_session(user_id):
    global scanning_session
    if user_id not in scanning_session:
        scanning_session[user_id] = {
            "counter": 0
        }
    else:
        scanning_session[user_id]["counter"] += 1

    for user_id, session in scanning_session.items():
        if session["counter"] < 50:
            continue

        add_today_attendence(user_id)
        scanning_session[user_id] = {
            "counter": 0
        }


def add_today_attendence(user_id):
    cur.execute(f"""SELECT *
                FROM attendance
                WHERE DATE(TIMESTAMP) = DATE() 
                LIMIT 1
                """)
    row = cur.fetchone()

    # Already timestamp today
    if row is not None:
        return

    cur.execute(
        """INSERT INTO attendance(user_id) VALUES ({})""".format(user_id))
    con.commit()
