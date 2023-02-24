import string
from pickle import dump, load
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical

# small library for seeing the progress of loops.
from tqdm import tqdm
tqdm.pandas()

# Loading a text file into memory
def loadText(file_name):
    # Opening the file as read only
    file=open(file_name, 'r')
    text=file.read() 
    file.close()
    return text

# get all imgs with their captions
def getImageCaptions(file_name):
    file=loadText(file_name)
    captions=file.split('\n')
    desc={}
    for caption in captions[:-1]:
        img, caption=caption.split('\t')
        if img[:-2] not in desc:
            desc[img[:-2]]=[caption]
        else:
            desc[img[:-2]].append(caption)
    return desc

# Data cleaning: converting to lower case, removing puntuations and words containing numbers
# A couple and an infant , being held by the male , sitting next to a pond with a near by stroller . 
# -> couple and an infant being held by the male sitting next to pond with near by stroller
def cleanCaptions(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,caption in enumerate(caps):
            caption.replace("-"," ")
            desc = caption.split()

            desc = [word.lower() for word in desc]                          # convertig to lower case
            desc = [word.translate(table) for word in desc]                 # removing punctuation from each token
            desc = [word for word in desc if(len(word)>1)]                  # removing hanging 's and a 
            desc = [word for word in desc if(word.isalpha())]               # removing tokens with numbers in them
            caption = ' '.join(desc)                                        # converting back to string
            captions[img][i]= caption

    return captions

# Separate all unique words and create the vocabulary from all the descriptions
def createVocabulary(descriptions):
    vocabulary=set()
    for key in descriptions.keys():
        [vocabulary.update(d.split()) for d in descriptions[key]]
    
    return vocabulary

# Store all captions in a file 
def storeCaptions(descriptions, file_name):
    lines=list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data="\n".join(lines)
    file=open(file_name,"w")
    file.write(data)
    file.close()

# Loading the text file in a string 
def loadPhotos(file_name):
    file=loadText(file_name)
    photos=file.split("\n")[:-1]
    return photos                           # Return the list of image names

# Loading clean captions
def loadCleanCaptions(file_name, photos):   
    file=loadText(file_name)
    captions={}                         # contains captions for each photo from the list of photos
    
    for line in file.split("\n"):
        str=line.split()
        
        if len(str)<1:
            continue
    
        image, image_caption=str[0], str[1:]
        
        if image in photos:
            if image not in captions:
                captions[image]=[]
            desc='<start> ' + " ".join(image_caption) + ' <end>'             # <start>, <end> are needed so that LSTM model can identify the starting and ending of the caption
            captions[image].append(desc)

    return captions

# Loading all features that have previously extracted from the Xception model
def loadFeatures(photos):
    all_features=load(open("features.p","rb"))
    features={k:all_features[k] for k in photos}                             # selecting only needed features
    return features

# Converting dictionary to clean list of captions
def dictToList(captions):
    all_cap=[]
    for key in captions.keys():
        [all_cap.append(d) for d in captions[key]]
    return all_cap

# Creating tokenizer class, mapping each word of the vocabulary with a unique index value
def createTokenizer(captions):
    cap_list=dictToList(captions)
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(cap_list)
    return tokenizer

#calculate maximum length of captions
def maxLength(captions):
    cap_list=dictToList(captions)
    return max(len(d.split()) for d in cap_list)