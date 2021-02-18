from flask import Flask, render_template, request
from keras.applications import ResNet50 
from keras.layers import Dense,LSTM,TimeDistributed,Embedding,Concatenate,RepeatVector,Activation
from keras.models import Sequential,Model 

from tqdm import tqdm 
from keras.preprocessing.sequence import pad_sequences 
import numpy as np
#resnet
resnet=ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
#model load 
vo=np.load('vocab.npy',allow_pickle=True)
vo=vo.item()
inv_vocab={v:k for k,v in vo.items()}
embedd_size=128  
max_len =40
vo_size=len(vo) 
langmod=Sequential() 
langmod.add(Embedding(input_dim=vo_size,output_dim=embedd_size,input_length=max_len))
langmod.add(LSTM(256,return_sequences=True))
langmod.add(TimeDistributed(Dense(embedd_size)))

igmod=Sequential() 
igmod.add(Dense(embedd_size,input_shape=(2048,),activation='relu'))
igmod.add(RepeatVector(max_len))

 
con=Concatenate()([igmod.output,langmod.output])

val=LSTM(128,return_sequences=True)(con)
val=LSTM(512,return_sequences=False)(val) 
val=Dense(vo_size)(val) 
out=Activation('softmax')(val)
model=Model(inputs=[igmod.input,langmod.input],outputs=out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.load_weights('model.h5')

app=Flask(__name__) 
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1 



@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['GET','POST'])
def predict(): 
    global model, vo, inv_vocab, resnet 
    file = request.files['file1']
    from cv2 import cv2
    file.save('static/file.jpg')
    img=cv2.imread('static/file.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(224,224)) 
    img=np.reshape(img,(1,224,224,3))
    features=resnet.predict(img).reshape(1,2048)
    text_in=['startofseq']
    final='' 
    
    count =0 
    while tqdm(count<20): 
        count+=1 
        encoded=[] 
        for i in text_in: 
            encoded.append(vo[i])
        padded=pad_sequences([encoded],maxlen=max_len,padding='post',truncating='post')
        sampled_index=np.argmax(model.predict([features,padded]))
        sampled_word=inv_vocab[sampled_index]

        if sampled_word!='endofseq':
            final=final+' '+sampled_word 

        text_in.append(sampled_word)

    return render_template('predict.html',final=final)
@app.route('/about')
def about():
    return render_template('about.html')
if __name__ == '__main__': 
    app.run(debug=True)