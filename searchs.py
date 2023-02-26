from keras.utils import pad_sequences
import numpy as np

def wordForId(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def generateCapGreedy(model, tokenizer, photo, max_length):
    in_text='start'
    
    for i in range(max_length):
        sequence=tokenizer.texts_to_sequences([in_text])[0]
        sequence=pad_sequences([sequence], maxlen=max_length)
        pred=model.predict([photo,sequence], verbose=0)
        pred=np.argmax(pred)
        word=wordForId(pred, tokenizer)
        
        if word is None:
            break
        
        in_text += ' ' + word
        
        if word=='end':
            break
    
    final=in_text.split()
    final=final[1:-1]
    final=' '.join(final)
    return final