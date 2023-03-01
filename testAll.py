from keras.utils import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
from pickle import load
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from helperFunctions import loadPhotos, loadCleanCaptions, loadFeatures
from searchs import generateCapGreedy, generateCapBeam

def corpus_meteor(expected, predicted):
    meteor_score_sentences_list=list()
    [meteor_score_sentences_list.append(meteor_score(expect, predict)) for expect, predict in zip(expected, predicted)]
    meteor_score_res=np.mean(meteor_score_sentences_list)
    return meteor_score_res

# Store all captions in a file 
def storeCaptionsPredicted(image_names, descriptions, file_name):
    lines=list()
    i=0
    for desc_list in descriptions:
        lines.append(image_names[i]+ '\t' +desc_list)
        i+=1
    data="\n".join(lines)
    file=open(file_name,"w")
    file.write(data)
    file.close()

def evaluateModelWithGreedySearch(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    listP=[]
    i=0
    print("Started...")
    for key, desc_list in descriptions.items():                                                     # Steping over the whole set
        yhat=generateCapGreedy(model, tokenizer, photos[key], max_length)                           # Generating description
        references=[d.split() for d in desc_list]                                                   # Storing actual and predicted
        actual.append(references)
        predicted.append(yhat.split())
        listP.append(yhat)
        i+=1
        if i%50==0:
            print("Image completed: ", i)

    storeCaptionsPredicted(test_images, listP, "./../captions_greedy_test_predicted.txt")

    # Calculating BLEU score
    print('BLEU-1 using greedy search: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2 using greedy search: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

    meteor_result=corpus_meteor(actual, predicted)
    print('Meteor score using greedy search: %f' % meteor_result)

def evaluateModelWithBeamSearch(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    listP=[]
    i=0
    print("Started...")
    for key, desc_list in descriptions.items():                                                     # Steping over the whole set
        yhat=generateCapBeam(model, tokenizer, photos[key], max_length, beam_index=3)               # Generating description
        references=[d.split() for d in desc_list]                                                   # Storing actual and predicted
        actual.append(references)
        predicted.append(yhat.split())
        listP.append(yhat)
        i+=1
        if i%50==0:
            print("Image completed: ", i)

    storeCaptionsPredicted(test_images, listP, "./../captions_beam_test_predicted.txt")

    # Calculating BLEU score
    print('BLEU-1 using beam search: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2 using beam search: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

    meteor_result=corpus_meteor(actual, predicted)
    print('Meteor score using beam search: %f' % meteor_result)

#--------------------------------------------------Main-----------------------------------------------------------------
# Preparing test set
test_images_path="./../Flickr8k_text/Flickr_8k.testImages.txt"
test_images=loadPhotos(test_images_path)

# Loading test set
test = loadPhotos(test_images_path)
print('Dataset: %d' % len(test))

# Captions
caption_path="./../captions.txt"
test_descriptions = loadCleanCaptions(caption_path, test)
print('Captions: test=%d' % len(test_descriptions))

# Photo features
test_features = loadFeatures(test)
print('Photos: test=%d' % len(test_features))

# Evaluating model
max_length=32
tokenizer_path="./../tokenizer.p"
tokenizer=load(open(tokenizer_path,"rb"))
model=load_model('./../model.h5')

#evaluateModelWithGreedySearch(model, test_descriptions, test_features, tokenizer, max_length)
evaluateModelWithBeamSearch(model, test_descriptions, test_features, tokenizer, max_length)