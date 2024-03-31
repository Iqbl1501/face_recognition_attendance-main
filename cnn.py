# Deep Learning CNN model to recognize face
'''This script uses a database of images and creates CNN model on top of it to test
   if the given image is recognized correctly or not'''

'''####### IMAGE PRE-PROCESSING for TRAINING and TESTING data #######'''

# Specifying the folder where images are present


import base64
import os
import sqlite3
import time
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Convolution2D
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
con = sqlite3.connect(os.environ["DATABASE_PATH"], check_same_thread=False)
cur = con.cursor()


def train_cnn():
    TrainingImagePath = 'dataset/'

    # Understand more about ImageDataGenerator at below link
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    # Defining pre-processing transformations on raw images of training data
    # These hyper parameters helps to generate slightly twisted versions
    # of the original image, which leads to a better model, since it learns
    # on the good and bad mix of images
    train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    # Defining pre-processing transformations on raw images of testing data
    # No transformations are done on the testing images
    test_datagen = ImageDataGenerator()

    # Generating the Training Data
    training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

    # Generating the Testing Data
    test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

    # =====================================

    '''############ Creating lookup table for all faces ############'''
    # class_indices have the numeric tag for each face
    TrainClasses = training_set.class_indices

    # Storing the face and the numeric tag for future reference
    for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
        cur.execute(f"SELECT * FROM users WHERE user_id = {faceValue}")
        rows = cur.fetchall()
        if len(rows) > 0:
            continue
        cur.execute(
            f"""INSERT INTO users(user_id, name) VALUES ({faceValue}, "{faceName}")""")
        con.commit()

    # Clear datset column
    cur.execute("DELETE FROM dataset")
    con.commit()
    # For dataset in db
    for idx, filepath in enumerate(training_set.filepaths):
        with open(filepath, "rb") as f:
            img_b64 = base64.b64encode(f.read())

            cur.execute(
                """INSERT INTO dataset(user_id, image) VALUES ({}, "{}")""".format(training_set.labels[idx], img_b64))
    con.commit()

    # The number of neurons for the output layer is equal to the number of faces
    OutputNeurons = len(TrainClasses)

    # # ==========================================
    '''######################## Create CNN deep learning model ########################'''

    '''Initializing the Convolutional Neural Network'''
    classifier = Sequential()

    ''' STEP--1 Convolution
    # Adding the first layer of CNN
    # we are using the format (64,64,3) because we are using TensorFlow backend
    # It means 3 matrix of size (64X64) pixels representing Red, Green and Blue components of pixels
    '''
    classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(
        1, 1), input_shape=(64, 64, 3), activation='relu'))

    '''# STEP--2 MAX Pooling'''
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    '''############## ADDITIONAL LAYER of CONVOLUTION for better accuracy #################'''
    classifier.add(Convolution2D(64, kernel_size=(
        5, 5), strides=(1, 1), activation='relu'))

    classifier.add(MaxPool2D(pool_size=(2, 2)))

    '''# STEP--3 FLattening'''
    classifier.add(Flatten())

    '''# STEP--4 Fully Connected Neural Network'''
    classifier.add(Dense(64, activation='relu'))

    classifier.add(Dense(OutputNeurons, activation='softmax'))

    '''# Compiling the CNN'''
    # classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.compile(loss='categorical_crossentropy',
                       optimizer='adam', metrics=["accuracy"])

    ###########################################################
    # Measuring the time taken by the model to train
    StartTime = time.time()

    # Starting the model training
    classifier.fit(
        training_set,
        steps_per_epoch=30,
        epochs=10,
        validation_data=test_set,
        validation_steps=10)

    EndTime = time.time()
    print("###### Total Time Taken: ", round(
        (EndTime-StartTime)/60), 'Minutes ######')

    # # =======================================
    # '''########### Saving Model ###########'''
    model_path = os.environ["CNN_PATH"]

    classifier.save(model_path)
    print(f'Saved model to: {model_path}')


if __name__ == "__main__":
    files_to_delete = ["model.keras", "ResultsMap.pkl"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
    train_cnn()
