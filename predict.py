import h5py
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

data_dir = "input/asl_alphabet_train"
target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 29
val_frac = 0.1
batch_size = 64

class ASLPredictor:
    def __init__(self):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        print("Loaded model from disk")
    
    def predict(self):
        image_dir = "input/asl_alphabet_test"
        # img = image.load_img(image_dir, target_size=target_size)
        # test_datagen = ImageDataGenerator(rescale=1./255)
        # test_generator = test_datagen.flow_from_directory(test_dir,target_size=target_size,color_mode="rgb",
        # shuffle = False,
        # class_mode='categorical',
        # batch_size=1)
        data_augmentor = ImageDataGenerator()
        train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True)
        label_map = (train_generator.class_indices) 
        print("Labels are ")
        print(label_map)
        
        test_generator = data_augmentor.flow_from_directory(image_dir, target_size=target_size, batch_size=batch_size, shuffle=False,class_mode='categorical')
        filenames = test_generator.filenames
        nb_samples = len(filenames) 
        print("testing : " + str(nb_samples))    
        probabilities = self.model.predict_generator(test_generator,steps=nb_samples,verbose=1)
        y_classes = probabilities.argmax(axis=-1)
        for x,y in zip(filenames,y_classes):
            print(x + "-> " + y)
    

c = ASLPredictor()
c.predict()
