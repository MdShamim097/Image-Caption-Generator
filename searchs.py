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

def generateCapBeam(model, tokenizer, photo, max_length, beam_index):
    start=tokenizer.texts_to_sequences(["start"])[0]
    start_word=[[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps=pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds=model.predict([photo,par_caps], verbose=0)
            word_preds=np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word=temp
        # Sorting according to the probabilities
        start_word=sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word=start_word[-beam_index:]
    
    start_word=start_word[-1][0]
    intermediate_caption=[wordForId(i, tokenizer) for i in start_word]
    final_caption=[]
    
    for i in intermediate_caption:
        if i != 'end':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption