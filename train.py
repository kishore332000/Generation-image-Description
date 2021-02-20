"""
Created on Thu Jan 28 13:41:08 2021

@author: kishorekannan
"""
import numpy as np
import cv2
from glob import glob
images_path='D:\\inframind\\Flicker8k_Dataset\\Images\\'
images=glob(images_path+'*.jpg')
#len(images)
from keras.applications import ResNet50
incept_model=ResNet50(include_top=False)
from keras.models import Model
last=incept_model.layers[-2].output
model=Model(inputs = incept_model.input,outputs=last)
images_features={}
count=0 
#image process
for i in images: 
    img=cv2.imread(i)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    img=cv2.resize(img,(224,224))
    img=img.reshape(1,224,224,3)
    pred=model.predict(img).reshape(2048,)
    img_name=i.split('\\')[-1]
    images_features[img_name]=pred
    count+=1 
#text processing
caption_path='D:\inframind\Flickr8k.token.txt'    
captions=open(caption_path,'rb').read().decode('utf-8').split('\n')
#create dict capitons
captions_dict={}
for i in captions:
    try:
        img_name=i.split('\t')[0][:-2]
        #print(img_name)
        caption=i.split('\t')[1]
        if img_name in images_features:
            if img_name not in captions_dict:
                captions_dict[img_name]=[caption]
            else:
                captions_dict[img_name].append(caption)
    except:
        pass
def perprocessed(txt):
    modified=txt.lower()
    modified='startofseq'+modified+'endofseq'
    return modified
for k,v in captions_dict.items():
    for vv in v:
        captions_dict[k][v.index(vv)]=perprocessed(vv)
#print(captions_dict)
count_words={}
count=1 
for k,vv in captions_dict.items():
    for v in vv:
        for word in v.split():
            if word not in count_words:
                count_words[word]=count 
                count+=1
#strings to integer 
for k, vv,in captions_dict.items():
    for v in vv:
        encoded=[]
        for word in v.split():
            encoded.append(count_words[word])
        captions_dict[k][vv.index(v)]=encoded
# creating captions arranging
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

MAX_LEN=0
for k, vv in captions_dict.items():
    for v in vv:
        if len(v)>MAX_LEN:
            MAX_LEN=len(v)
VOCAB_SIZE= len(count_words) 
def generator(photo, caption):
    x=[]
    y_in=[]
    y_out=[]
    for k,vv in caption.items():
        for v in vv:
            for i in range(1,len(v)):
                x.append(photo[k])
                in_seq=[v[:i]]
                out_seq=v[i]
                in_seq=pad_sequences(in_seq,maxlen=MAX_LEN,padding='post',truncating='post')[0]
                out_seq=to_categorical([out_seq],num_classes=VOCAB_SIZE+1)[0]
                y_in.append(in_seq)
                y_out.append(out_seq)
    #print(len(x),len(y_in),len(y_out))
    return x,y_in,y_out
x,y_in,y_out=generator(images_features,captions_dict)
#print(len(x),len(y_in),len(y_out))
x=np.array(x)
y_in=np.array(y_in,dtype='float64')
y_out=np.array(y_out,dtype='float64')
#model
from keras.models import Model, Sequential  
from keras.layers import Dense 
from keras.layers import LSTM 
from keras.layers import Embedding 
from keras.layers import TimeDistributed,Activation,RepeatVector,Concatenate
embedding_size=128 
max_len=MAX_LEN 
vocab_size=len(count_words)+1
image_model=Sequential() 
image_model.add(Dense(embedding_size,input_shape=(2048,),activation='relu'))
image_model.add(RepeatVector(max_len))
image_model.summary() 
language_model=Sequential() 
language_model.add(Embedding(input_dim=vocab_size,output_dim=embedding_size,input_length=max_len))
language_model.add(LSTM(256,return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca=Concatenate()([image_model.output,language_model.output])
x1=LSTM(128,return_sequences=True)(conca)
x1=LSTM(512,return_sequences=False)(x1)
x1=Dense(vocab_size)(x1) 
out=Activation('softmax')(x1)
model=Model(inputs=[image_model.input,language_model.input],outputs=out)

model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

# from keras.utils import plot_model 
# plot_model(model,show_shapes=True)
model.fit([x,y_in],y_out,batch_size=512,epochs=  50)
# creating model 
model.save('model.h5') 
model.save_weights('modweight.h5')
np.save('vocab.npy',count_words )

    