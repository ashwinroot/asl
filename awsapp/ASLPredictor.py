import h5py
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import os
import cv2

data_dir = "input/asl_alphabet_train"
target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 29
val_frac = 0.1
batch_size = 64

class ASLPredictor:
    def __init__(self):
        # load json and create model
        json_file = open('../.../models/model_self.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("models/model_self.h5")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        self.labels = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I','9': 'J', '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'W', '25': 'Z', '26': 'de', '27': 'nothing', '28': ' '}
        print("Loaded model from disk")
    
    def predict(self):
        image_dir = "input/asl_alphabet_test/test"
        for image_files in os.listdir(image_dir):
           image = cv2.imread(image_dir + '/' + image_files)
           if image is not None:
            img = np.asarray(np.resize(image,(1,64,64,3)))
            probabilities = self.model.predict(img)
            y_classes = probabilities.argmax(axis=-1)
            print(image_files + "-> "+str(y_classes))

    def predictor(self,image):
        img = np.asarray(np.resize(image,(1,64,64,3)))
        probabilities = self.model.predict(img)
        y_class = probabilities.argmax(axis=-1)[0]
        return str(self.labels[str(y_class)])
   

