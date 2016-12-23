# encoding: utf-8
#这里放置主程序以及IO
from numpy import *
from utils.tools import loadvoc
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Embedding
import threading
import time
import seq2seq
from seq2seq.models import AttentionSeq2Seq
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
VOC = len(ss) #最大词汇数目
SEN = 20 #句子最大长度

INPUT='' #输入的句子缓存
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
print('compiling...')
#输入层
#main_input = Input(shape=(SEN,), dtype='int32', name='main_input')
#x = Embedding(output_dim=VOC, input_dim=VOC, input_length=SEN)(main_input)
model = AttentionSeq2Seq(input_dim=VOC, hidden_dim=VOC, output_length=SEN, output_dim=VOC)
model.compile(loss='mse', optimizer='rmsprop')
print('model compiled..')

#模型训练-循环控制
POWER_OFF = False

def run():
    global INPUT,SPEAK_OUTPUT,POWER_OFF
    while not POWER_OFF:
        #读取输入数据进行训练
        if len(INPUT)>0:
            with rlock:
                X0 = s2i(INPUT)
            INPUT = ''
        else:
            with rlock:
                X0=s2i(SPEAK_OUTPUT)
            SPEAK_OUTPUT=''
        #读取系统时间
        X=zeros([1,SEN,VOC],dtype=int32)
        X[0]=to_categorical(X0[0],VOC)
        Y=X
        tm = time.localtime()
        X[0][-1] = tm.tm_hour
        model.fit(X,Y,
              nb_epoch=1, batch_size=1,verbose=1)
        
        SPEAK_OUTPUT=i2s(model.predict_classes(X,verbose=0))
        print('A: '+SPEAK_OUTPUT)
        time.sleep(1)

def say():
    global INPUT,SPEAK_OUTPUT,POWER_OFF
    while not POWER_OFF:
        a=raw_input('Q: ')
        if a == u'结束':
            POWER_OFF=a
        else:
            INPUT=a

        
threading.Thread(target = run, args = (), name = 'run').start()
threading.Thread(target = say, args = (), name = 'say').start() 

                 
                   
    
    
