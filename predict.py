import h5py
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.preprocessing import image

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
        image_dir = "input/asl_alphabet_test/A_test.jpg"
        img = image.load_img(image_dir, target_size=target_dims)
        print(img)
        # data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    # samplewise_std_normalization=True, 
                                    # validation_split=val_frac)
        # test_generator = data_augmentor.flow_from_directory(image_dir, target_size=target_size, batch_size=batch_size, shuffle=False)
        probabilities = self.model.predict(img)
    

c = ASLPredictor()
c.predict()
