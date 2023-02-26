from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from searchs import generateCapGreedy

ap=argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args=vars(ap.parse_args())
img_path=args['image']

def extractFeatures(filename, model):
        try:
            image=Image.open(filename)     
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")

        image=image.resize((299,299))
        image=np.array(image)
    
        if image.shape[2]==4:                             # for images that has 4 channels, we convert them into 3 channels
            image = image[..., :3]

        image=np.expand_dims(image, axis=0)
        image=image/255
        image=image - 1.0
        feature=model.predict(image)

        return feature

#--------------------------------------------------------Testing--------------------------------------------------------------    
max_length=32
path="./../"
tokenizer_file_name = path + "tokenizer.p"
tokenizer=load(open(tokenizer_file_name,"rb"))
model_file_name = path + "model.h5"
model=load_model(model_file_name)
inception_model=InceptionV3(include_top=False, pooling="avg")
photo=extractFeatures(img_path, inception_model)
image=Image.open(img_path)

print("\n")
print("Caption using Greedy Search: ", generateCapGreedy(model, tokenizer, photo, max_length))

plt.imshow(image)
plt.show()