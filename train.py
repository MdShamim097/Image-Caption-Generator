from pickle import dump, load
from helperFunctions import getImageCaptions, cleanCaptions, createVocabulary, storeCaptions
from helperFunctions import loadPhotos, loadCleanCaptions, loadFeatures, createTokenizer, maxLength
from CNN import extractFeatures
from LSTM import dataGenerator, defineModel

#--------------------------------------------------------------------------------------------------------------------------------
dataset_text = "C:/Users/user/Documents/CSE472/ImageCaptionGenerator/Flickr8k_text"
dataset_images = "C:/Users/user/Documents/CSE472/ImageCaptionGenerator/Flicker8k_Dataset"

# Preparing text data
file_name = dataset_text + "/" + "Flickr8k.token.txt"

#---------------------------------------------------Getting And Performing Data Cleaning-----------------------------------------
# Loading the file that contains all data
# Mapping them into captions dictionary img to 5 captions
captions=getImageCaptions(file_name)
print("Length of captions:" ,len(captions))

# Cleaning the captions
clean_captions=cleanCaptions(captions)

# Building vocabulary 
vocabulary=createVocabulary(clean_captions)
print("Length of vocabulary: ", len(vocabulary))

# Saving each caption to file 
storeCaptions(clean_captions, "captions.txt")

#---------------------------------------------------Extracting the Feature vector from all Images-------------------------------
# Dumping the features dictionary (2048 feature vector)
features = extractFeatures(dataset_images)
dump(features, open("features.p","wb"))

# Loading features disctionary 
features=load(open("features.p","rb"))

#--------------------------------------------------Loading Dataset For Training The Model---------------------------------------
file_name=dataset_text + "/" + "Flickr_8k.trainImages.txt"

test_images_path=dataset_text + "/" + "Flickr_8k.testImages.txt"
test_images=loadPhotos(test_images_path)

train_imgs=loadPhotos(file_name)
train_captions=loadCleanCaptions("captions.txt", train_imgs)
train_features=loadFeatures(train_imgs)

#--------------------------------------------------------Tokenizing the vocabulary----------------------------------------------
# Give each word a index, and store that into tokenizer.p pickle file
tokenizer=createTokenizer(train_captions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size=len(tokenizer.word_index) + 1
print("Number of words in vocabulary: ",vocab_size)

max_length_of_caption=maxLength(captions)
print("Maximum length of captions: ",max_length_of_caption)                         # 32

# #--------------------------------------------------------Testing Data Generator----------------------------------------------    
# # Input , Output of the Generator model
# [x1,x2],y = next(dataGenerator(train_captions, features, tokenizer, max_length_of_caption, vocab_size, 32)) 

# print("Shape of X1:", x1.shape)                           # (1754, 2048)
# print("Shape of X2:", x2.shape)                           # (1754, 32)
# print("Shape of y:", y.shape)                             # (1754, 7577)

#--------------------------------------------------------Training our model-----------------------------------------------
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_captions))
print('Photos: train=', len(train_features))
print("Shape of test_images: ", len(test_images))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length_of_caption)

model=defineModel(vocab_size, max_length_of_caption)
batch_size=32
epochs=15
steps=len(train_captions)//batch_size

for i in range(epochs):
    generator=dataGenerator(train_captions, train_features, tokenizer, max_length_of_caption, vocab_size, batch_size)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    if (i+1)==epochs:
        model.save("model.h5")