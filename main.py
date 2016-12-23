# encoding: utf-8
#这里放置主程序以及IO
from numpy import *
from utils.tools import loadvoc
from keras.models import Sequential,load_model,Model
from keras.layers import Input, Embedding, LSTM, Dense, merge, RepeatVector,TimeDistributed,Masking
from keras.optimizers import SGD,Adam
from keras.utils.np_utils import to_categorical
import threading
import time
rlock = threading.RLock()

#编码与解码文字
#i2c, c2i = loadvoc()
ss="1234567890-=qwertyuiopasdfghjkl;'zxcvbnm,."
i2c={}
c2i={}
for i in range(len(ss)):
    i2c[i]=ss[i]
    c2i[ss[i]]=i



#模型参数设置
VOC = len(i2c) #最大词汇数目
SEN = 20 #句子最大长度

INPUT=['',''] #输入的句子缓存
SPEAK_OUTPUT='' #输出的言语缓存

#将句子转化成数字
def s2i(s,l=SEN):
    idx=zeros([1,l],dtype=int32)
    for i in range(min(l,len(s))):
        idx[0,i]=c2i.get(s[i],0)
    return idx

def i2s(idx):
    
    s=''
    for i in idx[0,:]:
        if i>0:
            s.join(i2c.get(i,''))
    return s

#定义主模型
#输入层
main_input = Input(shape=(SEN,), dtype='int32', name='main_input')
#文字矢量化层
x = Masking(mask_value=0)(main_input)
x = Embedding(output_dim=VOC, input_dim=VOC, input_length=SEN)(x)
#长短记忆层
lstm_out = LSTM(128)(x)

#生物钟，当前时间信息输入[hr,min]
time_input = Input(shape=(2,), name='time_input')
#生物钟激活函数
time_out = Dense(128, activation='sigmoid')(time_input)
#生物钟作为阀门
x = merge([lstm_out, time_out], mode='mul')

# 语言逻辑深层网络
x = Dense(128, activation='relu')(x)
# 时序言语输出
x = RepeatVector(SEN)(x)
speak_output = TimeDistributed(Dense(VOC, activation='sigmoid'),name='speak_output')(x)
#speak_output = LSTM(VOC,activation='softmax', name='speak_output',return_sequences=True)(x)

# 模型封装
model = Sequential()
model.add(Model(input=[main_input, time_input], output=speak_output))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#模型训练-循环控制
POWER_OFF = False

def run():
    global INPUT,SPEAK_OUTPUT,POWER_OFF
    while not POWER_OFF:
        #读取输入数据进行训练
        if len(INPUT[0]) == 0:
            with rlock:
                INPUT[1] = INPUT[0]
                INPUT[0] = SPEAK_OUTPUT
        X = s2i(INPUT[1])
        Y = s2i(INPUT[0])
        #读取系统时间
        tm = time.localtime()
        TIME_INPUT = asarray([[tm.tm_hour,tm.tm_min]],dtype=int32)
        Y=zeros([1,SEN,VOC],dtype=int32)
        Y[0]=to_categorical(X[0],VOC)
        model.fit([X, TIME_INPUT],Y,
              nb_epoch=1, batch_size=1,verbose=0)
        
        SPEAK_OUTPUT=i2s(model.predict_classes([X,TIME_INPUT],verbose=0))
        if len(SPEAK_OUTPUT)>0:
           print('A: '+SPEAK_OUTPUT)
        time.sleep(1)

def say():
    global INPUT,SPEAK_OUTPUT,POWER_OFF
    while not POWER_OFF:
        a=raw_input('Q: ')
        if a == u'end':
            POWER_OFF = a
        else:
            INPUT[1] = INPUT[0]
            INPUT[0] = a

        
threading.Thread(target = run, args = (), name = 'run').start()
threading.Thread(target = say, args = (), name = 'say').start() 

                 
                   
    
    
