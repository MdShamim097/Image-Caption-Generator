import numpy as np
from keras.utils import pad_sequences, to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merging import add
from keras.utils import plot_model

# Data generator, used by model.fit_generator()
def dataGenerator(descriptions, features, tokenizer, max_length, vocab_size, num_photos_per_batch):
    X1, X2, y=list(), list(), list()
    n=0
    while 1:
        for image_name, description_list in descriptions.items():
            n+=1
            feature=features[image_name][0]                                                 # retrieving photo features
        
            for desc in description_list:                                                   # walking through each description for the image
                seq=tokenizer.texts_to_sequences([desc])[0]                                 # encoding the sequence
                for i in range(1, len(seq)):                                                # spliting one sequence into multiple X,y pairs
                    in_seq, out_seq=seq[:i], seq[i]                                         # spliting into input and output pair
                    in_seq=pad_sequences([in_seq], maxlen=max_length)[0]                    # padding input sequence
                    out_seq=to_categorical([out_seq], num_classes=vocab_size)[0]            # encoding output sequence
                    X1.append(feature)                                                      # storing
                    X2.append(in_seq)
                    y.append(out_seq)
            
            if n==num_photos_per_batch:
                input_image, input_sequence, output_word=np.array(X1), np.array(X2), np.array(y)
                yield[[input_image, input_sequence], output_word]  
                X1, X2, y=list(), list(), list()
                n=0 

# Part-1 Photo feature extractor - we extracted features from pretrained model Inception_v3. 
# Part-2 Sequence processor - word embedding layer that handles text, followed by LSTM 
# Part-3 Decoder - Both 1 and 2 model produce fixed length vector. They are merged together and processed by dense layer to make final prediction

# Defining captioning model
def defineModel(vocab_size, max_length):
    inputs1=Input(shape=(2048,))                                          # features from the CNN model squeezed from 2048 to 512 nodes
    pfe1=Dropout(0.5)(inputs1)
    pfe2=Dense(512, activation='relu')(pfe1)

    # LSTM sequence model
    inputs2=Input(shape=(max_length,))
    sp1=Embedding(vocab_size, 512, mask_zero=True)(inputs2)
    sp2=Dropout(0.5)(sp1)
    sp3=LSTM(512)(sp2)

    # Merging both models
    dec1=add([pfe2, sp3])
    dec2=Dense(512, activation='relu')(dec1)
    outputs=Dense(vocab_size, activation='softmax')(dec2)
    
    # tie it together [image_feature, seq] [word]
    model=Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # summarizing model
    print(model.summary())
    
    return model