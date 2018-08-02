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
        json_file = open('models/self/model_self.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("models/self/model_self.h5")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        self.labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
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
        return str(self.labels[y_class])
   

