# encoding: utf-8
#这里放置主程序以及IO
from numpy import *
from utils.tools import loadvoc
from keras.models import Sequential,load_model,Model
from keras.layers import Input, Reshape,Embedding, LSTM, Dense, merge, RepeatVector,TimeDistributed,Masking,Activation
from keras.optimizers import SGD,Adam
from keras.utils.np_utils import to_categorical
import threading
import time
import os
rlock = threading.RLock()

#编码与解码文字
#i2c, c2i = loadvoc()
ss="qwertyuiopasdfghjkl'zxcvbnm,.-?! "
i2c={}
c2i={}
for i in range(len(ss)):
    i2c[i+1]=ss[i]
    c2i[ss[i]]=i+1



#模型参数设置
VOC = len(i2c) #最大词汇数目
SEN = 100 #句子最大长度
M = 20 # 短期记忆
INPUT=['' for x in range(M)] #输入的句子缓存
SPEAK_OUTPUT='' #输出的言语缓存

def store(s,M):
    if not (s == M[0] and s == M[1]):
        M1=[s]
        M1.extend(M[0:-1])
        return M1
    else:
        return M
def get(M):
    L=len(M)-mod(len(M),2)
    MY=[]
    MX=[]
    for i in range(0,L,2):
        MX.append(M[i+1])
        MY.append(M[i])
    return MX,MY

#将句子转化成数字
def s2i(S,SEN=SEN):
    N=len(S)
    idx=zeros([N,SEN,1],dtype=int32)
    for n in range(N):
        s=S[n]
        for i in range(min(SEN,len(s))):
            idx[n,i,0]=c2i.get(s[i],0)
    return idx

def i2s(idx):
    N=len(idx)
    S=[]
    for n in range(N):
        s=''
        for i in idx[n,:,0]:
            if i>0:
                s+=i2c.get(int(round(i)),'')
        S.append(s)
    return S

#定义主模型
model = Sequential()
model.add(LSTM(input_dim=1, output_dim=64, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(output_dim=SEN))
model.add(Reshape((SEN,1)))
model.add(Activation('linear'))
print('compiling...')
model.compile(loss='mse', optimizer='rmsprop')
print('compiled')
#模型训练-循环控制
POWER_OFF = False
SPEAK=False
def run():
    global INPUT,SPEAK_OUTPUT,POWER_OFF,SPEAK
    while not POWER_OFF:
        #读取输入数据进行训练
        X,Y=get(INPUT)
        X=s2i(X)
        Y=s2i(Y)
#        print('thinking...')
        model.fit(X,Y,
              nb_epoch=1, batch_size=len(X),verbose=0)
        yy=model.predict(X[0:1],verbose=0)
        SPEAK_OUTPUT=i2s(yy)
        if SPEAK:
           print('\nA: '+SPEAK_OUTPUT[0]+'\n')
           SPEAK=False
        time.sleep(0.5)

def say():
    global INPUT,SPEAK_OUTPUT,POWER_OFF,SPEAK
    while not POWER_OFF:
        if not SPEAK:
            a=raw_input('Q: ').lower()
        if a == u'end':
            POWER_OFF = a
            model.save('baby-model.h5')
        else:
            SPEAK=True
            INPUT=store(a,INPUT)

        
threading.Thread(target = run, args = (), name = 'run').start()
threading.Thread(target = say, args = (), name = 'say').start() 

                 
                   
    
    
