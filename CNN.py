import numpy as np
from PIL import Image
import os
from pickle import dump, load
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.utils import load_img, img_to_array
from keras.utils import pad_sequences, to_categorical

# small library for seeing the progress of loops.
from tqdm import tqdm
tqdm.pandas()

# Extract features for all images and map image names with their  respective feature array
def extractFeatures(directory):
        model=InceptionV3(include_top=False, pooling='avg')
        features={}
        for image_name in tqdm(os.listdir(directory)):
            filename=directory + "/" + image_name
            image=Image.open(filename)
            image=image.resize((299,299))
            image=np.expand_dims(image, axis=0)         # Inserting a new axis that will appear at the "axis" position in the expanded array shape.
            image=image/255                             # Normalizing
            image=image-1.0
            feature=model.predict(image)
            features[image_name]=feature
        return features