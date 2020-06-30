import numpy as np #line:1
import pprint as pp #line:2
import tensorflow as tf #line:3
import math #line:4
from keras .models import Sequential #line:5
from keras .layers import Dense ,Dropout ,Activation ,Flatten ,Multiply ,Reshape #line:6
from keras .layers import Conv1D ,MaxPooling1D ,GlobalAveragePooling1D ,AveragePooling1D #line:7
from keras import backend as K #line:8
from keras .optimizers import SGD #line:9
from keras import regularizers #line:10
from keras .models import Model #line:11
from keras .models import load_model #line:12
import keras #line:13
from keras .applications .vgg16 import preprocess_input ,decode_predictions #line:14
import scipy .interpolate as spi #line:15
import matplotlib .pyplot as plt #line:16
import seaborn as sns #line:17
from matplotlib .collections import LineCollection #line:18
from matplotlib .colors import ListedColormap ,BoundaryNorm #line:19
from keras .callbacks import ModelCheckpoint #line:20
from keras .layers .core import Lambda #line:21
from tensorflow .python .framework import ops #line:22
config =tf .ConfigProto ()#line:23
config .gpu_options .allow_growth =True #line:24
from sklearn .model_selection import validation_curve #line:25
from sklearn import svm #line:26
from sklearn .model_selection import cross_val_score #line:27
import scipy .io as io #line:28
from sklearn .model_selection import train_test_split #line:29
config =tf .ConfigProto ()#line:30
config .gpu_options .allow_growth =True #line:31
class_n =9 #line:32
def allPreprocessingOperation ():#line:33
    OO0OO0OOO0OOOO0OO =io .loadmat ("PaviaU.mat")#line:34
    OOO000OO00O00O0OO =io .loadmat ("PaviaU_gt.mat")#line:35
    OO0OO0OOO0OOOO0OO =OO0OO0OOO0OOOO0OO ['paviaU']#line:36
    OOO000OO00O00O0OO =OOO000OO00O00O0OO ['paviaU_gt']#line:37
    np .save ('paviaU.npy',OO0OO0OOO0OOOO0OO )#line:38
    np .save ('paviaU_gt.npy',OOO000OO00O00O0OO )#line:39
    OO0OO0OOO0OOOO0OO =np .load ('paviaU.npy')#line:40
    OOO000OO00O00O0OO =np .load ('paviaU_gt.npy')#line:41
    OO0OO0OOO0OOOO0OO =OO0OO0OOO0OOOO0OO .reshape (OO0OO0OOO0OOOO0OO .shape [0 ]*OO0OO0OOO0OOOO0OO .shape [1 ],OO0OO0OOO0OOOO0OO .shape [2 ])#line:42
    OOO000OO00O00O0OO =OOO000OO00O00O0OO .reshape (OOO000OO00O00O0OO .shape [0 ]*OOO000OO00O00O0OO .shape [1 ])#line:43
    OO0OO0OOO0OOOO0OO =OO0OO0OOO0OOOO0OO /OO0OO0OOO0OOOO0OO .max ()#line:44
    OOO000OO00O00O0OO =OOO000OO00O00O0OO -1 #line:45
    return OO0OO0OOO0OOOO0OO ,OOO000OO00O00O0OO #line:46
def preprocessingData (O00000O0O0OOOOO00 ,OO0O0000O00000000 ,OOOO0O0O00O0O00OO ):#line:47
    O00O00O0O0O0O0O00 =[]#line:48
    O00O00O0O00OOOOO0 =[]#line:49
    OO0OOO000OO00O0O0 =[]#line:50
    OOO0OO0OOOO0O0O0O =[]#line:51
    for O000O00O0O000OO0O in range (class_n ):#line:52
        O00O0O0O0O0O0O0OO =np .where (OO0O0000O00000000 ==O000O00O0O000OO0O )#line:53
        O00O0O0O0O0O0O0OO =O00O0O0O0O0O0O0OO [0 ]#line:54
        OO000O00OO0OO00OO =O00000O0O0OOOOO00 [O00O0O0O0O0O0O0OO ]#line:55
        OO0OOOOO0OOOO0OO0 =OO0O0000O00000000 [O00O0O0O0O0O0O0OO ]#line:56
        O00OO000OO0O0000O ,O0O0O0O0O000O000O ,OO0O0O0000O0OO0OO ,OO0OO0O0O0OO0O0OO =train_test_split (OO000O00OO0OO00OO ,OO0OOOOO0OOOO0OO0 ,train_size =OOOO0O0O00O0O00OO ,random_state =42 )#line:57
        O00O00O0O0O0O0O00 .append (O00OO000OO0O0000O )#line:58
        OO0OOO000OO00O0O0 .append (O0O0O0O0O000O000O )#line:59
        O00O00O0O00OOOOO0 .append (OO0O0O0000O0OO0OO )#line:60
        OOO0OO0OOOO0O0O0O .append (OO0OO0O0O0OO0O0OO )#line:61
    OOOOO0000O0O0O00O =np .array (O00O00O0O0O0O0O00 [0 ])#line:62
    OOOOOOOO000O0O0OO =np .array (OO0OOO000OO00O0O0 [0 ])#line:63
    O00OOO00OOOO00OOO =np .array (O00O00O0O00OOOOO0 [0 ])#line:64
    OO0OOOOO0O00O00O0 =np .array (OOO0OO0OOOO0O0O0O [0 ])#line:65
    for O000O00O0O000OO0O in range (class_n -1 ):#line:66
        OOOOO0000O0O0O00O =np .vstack ((OOOOO0000O0O0O00O ,np .array (O00O00O0O0O0O0O00 [O000O00O0O000OO0O +1 ])))#line:67
        OOOOOOOO000O0O0OO =np .vstack ((OOOOOOOO000O0O0OO ,np .array (OO0OOO000OO00O0O0 [O000O00O0O000OO0O +1 ])))#line:68
        O00OOO00OOOO00OOO =np .hstack ((O00OOO00OOOO00OOO ,np .array (O00O00O0O00OOOOO0 [O000O00O0O000OO0O +1 ])))#line:69
        OO0OOOOO0O00O00O0 =np .hstack ((OO0OOOOO0O00O00O0 ,np .array (OOO0OO0OOOO0O0O0O [O000O00O0O000OO0O +1 ])))#line:70
    O0O00O0O00O0O0O0O =np .random .choice (O00OOO00OOOO00OOO .shape [0 ],O00OOO00OOOO00OOO .shape [0 ],replace =False )#line:71
    OOOOO0000O0O0O00O =OOOOO0000O0O0O00O [O0O00O0O00O0O0O0O ]#line:72
    O00OOO00OOOO00OOO =O00OOO00OOOO00OOO [O0O00O0O00O0O0O0O ]#line:73
    OOOO0O0O00O0O00OO =int (OOOO0O0O00O0O00OO *100 )#line:74
    OO0O00OOO0OO00O0O ='train_data_'+str (OOOO0O0O00O0O00OO )+'.npy'#line:75
    O00000OO0O00OOOO0 ='train_label_'+str (OOOO0O0O00O0O00OO )+'.npy'#line:76
    O00O0O00OO000O0O0 ='test_data_'+str (100 -OOOO0O0O00O0O00OO )+'.npy'#line:77
    OO0O00O0000O0O0OO ='test_label_'+str (100 -OOOO0O0O00O0O00OO )+'.npy'#line:78
    np .save (OO0O00OOO0OO00O0O ,OOOOO0000O0O0O00O )#line:79
    np .save (O00000OO0O00OOOO0 ,O00OOO00OOOO00OOO )#line:80
    np .save (O00O0O00OO000O0O0 ,OOOOOOOO000O0O0OO )#line:81
    np .save (OO0O00O0000O0O0OO ,OO0OOOOO0O00O00O0 )#line:82
    return OOOOO0000O0O0O00O ,OOOOOOOO000O0O0OO ,O00OOO00OOOO00OOO ,OO0OOOOO0O00O00O0 #line:83
def loadTrainData (n =0 ):#line:84
    if n ==0 :#line:85
        O000OO00OO0OO00O0 =np .load ('train_data.npy')#line:86
        O0O00O0OO00O0000O =np .load ('train_label.npy')#line:87
        OO0O0000OO0OOOOOO =np .load ('test_data.npy')#line:88
        OOOO00O0OO0OOOO00 =np .load ('test_label.npy')#line:89
    else :#line:90
        O000OO00OO0OO00O0 ='train_data_'+str (n )+'.npy'#line:91
        O0O00O0OO00O0000O ='train_label_'+str (n )+'.npy'#line:92
        OO0O0000OO0OOOOOO ='test_data_'+str (100 -n )+'.npy'#line:93
        OOOO00O0OO0OOOO00 ='test_label_'+str (100 -n )+'.npy'#line:94
        O000OO00OO0OO00O0 =np .load (O000OO00OO0OO00O0 )#line:95
        O0O00O0OO00O0000O =np .load (O0O00O0OO00O0000O )#line:96
        OO0O0000OO0OOOOOO =np .load (OO0O0000OO0OOOOOO )#line:97
        OOOO00O0OO0OOOO00 =np .load (OOOO00O0OO0OOOO00 )#line:98
    O000OO00OO0OO00O0 =O000OO00OO0OO00O0 .reshape (O000OO00OO0OO00O0 .shape [0 ],O000OO00OO0OO00O0 .shape [1 ],1 )#line:99
    OO0O0000OO0OOOOOO =OO0O0000OO0OOOOOO .reshape (OO0O0000OO0OOOOOO .shape [0 ],OO0O0000OO0OOOOOO .shape [1 ],1 )#line:100
    O0O00O0OO00O0000O =one_hot (O0O00O0OO00O0000O ,n_class =9 )#line:101
    OOOO00O0OO0OOOO00 =one_hot (OOOO00O0OO0OOOO00 ,n_class =9 )#line:102
    return O000OO00OO0OO00O0 ,OO0O0000OO0OOOOOO ,O0O00O0OO00O0000O ,OOOO00O0OO0OOOO00 #line:103
def one_hot (OO00OOO00O0OO00OO ,n_class =9 ):#line:104
    OO00OOO00O0OO00OO =tf .one_hot (OO00OOO00O0OO00OO ,n_class ,1 ,0 )#line:105
    with tf .Session (config =config )as O0O0OOO0OOO00O0OO :#line:106
        OO00OOO00O0OO00OO =O0O0OOO0OOO00O0OO .run (OO00OOO00O0OO00OO )#line:107
    return OO00OOO00O0OO00OO #line:108
def creatTrainData ():#line:109
    OO0O00O0O000O00OO ,OO0O0O0OO0O000O0O =allPreprocessingOperation ()#line:110
    O0O00OO0OO00O0O00 ,OOOO0O0000OOOOO0O ,OO000OOOO000OO00O ,O0000OO0O0O0O0O00 =preprocessingData (OO0O00O0O000O00OO ,OO0O0O0OO0O000O0O ,0.01 )#line:111
import cv2 #line:112
import sys #line:113
import numpy as np #line:114
creatTrainData ()#line:115
train_data ,test_data ,train_label ,test_label =loadTrainData (1 )#line:116
def train_model (O0OOOO0000O0O00OO ,OOOOOOO0OOO000OO0 ,OO000OOO0OO000O0O ,OO0O0OOOO0O000000 ,OO000O0O0O00O000O ,n =100 ,ep =1000 ):#line:117
    OOO0OOOOO0OO00OOO ="weights.best.hdf5"#line:118
    OO0000OOOO0O000O0 =ModelCheckpoint (OOO0OOOOO0OO00OOO ,monitor ='val_acc',verbose =1 ,save_best_only =True ,mode ='max')#line:119
    OOOO000OO00000O0O =[OO0000OOOO0O000O0 ]#line:120
    O000O00O00O000000 =O0OOOO0000O0O00OO .fit (OOOOOOO0OOO000OO0 ,OO000OOO0OO000O0O ,batch_size =n ,epochs =ep ,verbose =1 ,validation_data =(OO0O0OOOO0O000000 ,OO000O0O0O00O000O ),callbacks =OOOO000OO00000O0O )#line:121
    O0OOOO0000O0O00OO .save ('model.h5')#line:122
    OOOOO000O0OOOOOO0 =O0OOOO0000O0O00OO .evaluate (OO0O0OOOO0O000000 ,OO000O0O0O00O000O ,verbose =0 )#line:123
    print ('Test score:',OOOOO000O0OOOOOO0 [0 ])#line:124
    print ('Test accuracy:',OOOOO000O0OOOOOO0 [1 ])#line:125
def selectOneBand (OOOOO00OO00OOO000 ,O0OO0OOOO0O0OOOOO ,n =0 ):#line:126
    return OOOOO00OO00OOO000 [n :n +1 ],O0OO0OOOO0O0OOOOO [n :n +1 ]#line:127
def normalize (OO0OOO0OO0O00O0OO ):#line:128
    print (OO0OOO0OO0O00O0OO /(math .sqrt (math .mean (math .square (OO0OOO0OO0O00O0OO )))+1e-5 ))#line:129
def one_hot (OOOO0O0000OOO000O ,n_class =9 ):#line:130
    OOOO0O0000OOO000O =tf .one_hot (OOOO0O0000OOO000O ,n_class ,1 ,0 )#line:131
    with tf .Session (config =config )as O0OO0OOO0OOOOO00O :#line:132
        OOOO0O0000OOO000O =O0OO0OOO0OOOOO00O .run (OOOO0O0000OOO000O )#line:133
    return OOOO0O0000OOO000O #line:134
def find_one_class (OO000OO00O000O000 ,O00OO0OO0OO0OOO00 ,O0OOO0O00O0O0OO0O ,one_class =0 ):#line:135
    O000O00000OO00OOO =OO000OO00O000O000 .predict_classes (O00OO0OO0OO0OOO00 )#line:136
    O000O00000OO00OOO =one_hot (O000O00000OO00OOO )#line:137
    one_class =one_hot (one_class )#line:138
    OO0O0O000O0OO0000 =(O0OOO0O00O0O0OO0O *O000O00000OO00OOO *one_class ).sum (axis =1 )#line:139
    O00OOOOOOO00OO0OO =np .array (np .where (OO0O0O000O0OO0000 ==1 )).reshape (-1 )#line:140
    OOO0OOO0OO000OO0O =O00OO0OO0OO0OOO00 [O00OOOOOOO00OO0OO ]#line:141
    OOOO0OOOOO00O00O0 =O0OOO0O00O0O0OO0O [O00OOOOOOO00OO0OO ]#line:142
    return OOO0OOO0OO000OO0O ,OOOO0OOOOO00O00O0 #line:143
def creatModelTest ():#line:144
    OO000OOOO0O0OOO00 =Sequential ()#line:145
    OO000OOOO0O0OOO00 .add (Conv1D (filters =32 ,kernel_size =3 ,padding ='same',input_shape =(103 ,1 ),activation ='relu',name ="C1"))#line:146
    OO000OOOO0O0OOO00 .add (Conv1D (filters =32 ,kernel_size =3 ,padding ='same',activation ='relu',name ='C2'))#line:147
    OO000OOOO0O0OOO00 .add (MaxPooling1D (pool_size =2 ,name ='P2'))#line:148
    OO000OOOO0O0OOO00 .add (Conv1D (filters =64 ,kernel_size =3 ,padding ='same',activation ='relu',name ='C3'))#line:149
    OO000OOOO0O0OOO00 .add (Conv1D (filters =128 ,kernel_size =3 ,padding ='same',activation ='relu',name ='C4'))#line:150
    OO000OOOO0O0OOO00 .add (Flatten ())#line:151
    OO000OOOO0O0OOO00 .add (Dropout (0.25 ))#line:152
    OO000OOOO0O0OOO00 .add (Dense (9 ,activation ='softmax'))#line:153
    OOO0OO00OOOO0OO0O =SGD (lr =0.01 ,decay =1e-7 ,momentum =0.9 ,nesterov =True )#line:154
    OO000OOOO0O0OOO00 .compile (loss ='categorical_crossentropy',optimizer =OOO0OO00OOOO0OO0O ,metrics =['accuracy'])#line:155
    OO000OOOO0O0OOO00 .summary ()#line:156
    return OO000OOOO0O0OOO00 #line:157
def register_gradient ():#line:158
    if "GuidedBackProp"not in ops ._gradient_registry ._registry :#line:159
        @ops .RegisterGradient ("GuidedBackProp")#line:160
        def _OOO0O000OOO0O0OO0 (OOOOOO0OO000O00OO ,OO0O000000000O0O0 ):#line:161
            O0O00O00000000O00 =OOOOOO0OO000O00OO .inputs [0 ].dtype #line:162
            return OO0O000000000O0O0 *tf .cast (OO0O000000000O0O0 >0 ,O0O00O00000000O00 )#line:163
def modify_backprop (OO0O0OOO0O000OO0O ,OO0O00OOOO0O00O0O ):#line:164
    OO0OOO00000O000O0 =tf .get_default_graph ()#line:165
    with OO0OOO00000O000O0 .gradient_override_map ({'Relu':OO0O00OOOO0O00O0O }):#line:166
        OOOO00O000O0OOOO0 =[OO00OOOOO0O000OO0 for OO00OOOOO0O000OO0 in OO0O0OOO0O000OO0O .layers [0 :]if hasattr (OO00OOOOO0O000OO0 ,'activation')]#line:167
        for O00O00O0O000OOO0O in OOOO00O000O0OOOO0 :#line:168
            if O00O00O0O000OOO0O .activation ==keras .activations .relu :#line:169
                O00O00O0O000OOO0O .activation =tf .nn .relu #line:170
        O00OOO00OO0000OO0 =creatModelTest ()#line:171
        O00OOO00OO0000OO0 .load_weights ('weights.best.hdf5')#line:172
    return O00OOO00OO0000OO0 #line:173
def compile_saliency_function (O00O0O0OO0OOOO0OO ,activation_layer ='C4'):#line:174
    O0000OO0OOOO00O0O =O00O0O0OO0OOOO0OO .input #line:175
    O0OOOOOOOOOOO00O0 =dict ([(OO0O0O00OO00O0OOO .name ,OO0O0O00OO00O0OOO )for OO0O0O00OO00O0OOO in O00O0O0OO0OOOO0OO .layers [0 :]])#line:176
    OOO00O0OO00O0O0O0 =O0OOOOOOOOOOO00O0 [activation_layer ].output #line:177
    OO0OO00OO0O000O00 =K .max (OOO00O0OO00O0O0O0 ,axis =2 )#line:178
    OO0OO00OO0O000O00 ,O0O00000OO0O000O0 =K .tf .nn .top_k (OO0OO00OO0O000O00 ,k =30 )#line:179
    O0O0OOO0O0OO0O0OO =K .gradients (K .sum (OO0OO00OO0O000O00 ),O0000OO0OOOO00O0O )[0 ]#line:180
    return K .function ([O0000OO0OOOO00O0O ,K .learning_phase ()],[O0O0OOO0O0OO0O0OO ])#line:181
def bandSelectionAverage (OO0OOO0OOO0O00O00 ,O0OOOO00OO000O0OO ,O000OOOO00OO00O00 ,OO000000O00OO0OOO ,O000O0O0OOOOOOOOO ,O00OOO00O000O0OO0 ,OOO0000O0O0OOOO0O ):#line:182
    O00OOO00O000O000O =math .ceil (O00OOO00O000O0OO0 /OOO0000O0O0OOOO0O )#line:183
    O00000OOOO000O00O =np .zeros (OO0OOO0OOO0O00O00 .shape [1 ])#line:184
    O00OO0OO0O0O00OOO =np .array ([6631 ,18649 ,2099 ,3064 ,1345 ,5029 ,1330 ,3682 ,947 ])#line:185
    OO0OO0O000OO0O0OO =0 #line:186
    for O0O00O0O0OO00OOOO in (O00OO0OO0O0O00OOO ).argsort ():#line:187
        O0000OO0OOO00O00O =O000O0O0OOOOOOOOO [O0O00O0O0OO00OOOO ]#line:188
        O0OOO0O0OO0O0O000 =np .argsort (-O0000OO0OOO00O00O )#line:189
        OOO00OO000O0OOO00 =0 #line:190
        while (OOO00OO000O0OOO00 <O00OOO00O000O000O ):#line:191
            O00OOOOOO00OOO0O0 =OOO00OO000O0OOO00 #line:192
            while (O00000OOOO000O00O [O0OOO0O0OO0O0O000 [O00OOOOOO00OOO0O0 ]]==1 ):#line:193
                O00OOOOOO00OOO0O0 =O00OOOOOO00OOO0O0 +1 #line:194
            O00000OOOO000O00O [O0OOO0O0OO0O0O000 [O00OOOOOO00OOO0O0 ]]=1 #line:195
            OO0OO0O000OO0O0OO =OO0OO0O000OO0O0OO +1 #line:196
            OOO00OO000O0OOO00 =OOO00OO000O0OOO00 +1 #line:197
            if OO0OO0O000OO0O0OO ==O00OOO00O000O0OO0 :break #line:198
        if OO0OO0O000OO0O0OO ==O00OOO00O000O0OO0 :break #line:199
    O00O00OOOO00OO000 =np .where (O00000OOOO000O00O ==1 )[0 ]#line:200
    O000000OOOO00O000 =np .squeeze (OO0OOO0OOO0O00O00 [:,O00O00OOOO00OO000 ,:])#line:201
    OO00O00OOO00OOOO0 =np .squeeze (O0OOOO00OO000O0OO [:,O00O00OOOO00OO000 ,:])#line:202
    OO0000O0O00OOOO00 =backOneHot (O000OOOO00OO00O00 )#line:203
    OO000000O00OO0OOO =backOneHot (OO000000O00OO0OOO )#line:204
    return O000000OOOO00O000 ,OO00O00OOO00OOOO0 ,O00000OOOO000O00O ,OO0000O0O00OOOO00 ,OO000000O00OO0OOO #line:205
def get_one_result (O0O0O00OOOOOO00O0 ,O000OO000000O00OO ,n =100 ,plt_class =0 ):#line:206
    OO0O000O00OOO00OO ,O0O00OOO0OOOO0OO0 =find_one_class (O000OO000000O00OO ,train_data ,train_label ,one_class =plt_class )#line:207
    OO00O000000OOO00O =OO0O000O00OOO00OO [0 :n ]#line:208
    OOOOOOOO0O0O0O000 =np .squeeze (O0O0O00OOOOOO00O0 ([OO00O000000OOO00O ,0 ])[0 ])#line:209
    return OOOOOOOO0O0O0O000 #line:210
def get_all_result (m =9 ,name ='C1'):#line:211
    O0OOOOO0O0000000O =[]#line:212
    OO0OOOO0OOO0O0000 =[]#line:213
    O000000O00O0O000O ,O00OO0O0OO00OOOO0 =alter_model (layer_name =name )#line:214
    for OOO0OO0O00OO0OOO0 in range (m ):#line:215
        O0OOO00OO000OOO0O =get_one_result (O000000O00O0O000O ,O00OO0O0OO00OOOO0 ,30 ,plt_class =OOO0OO0O00OO0OOO0 )#line:216
        OOO0O000O0OO0OO0O =np .maximum (np .mean (O0OOO00OO000OOO0O ,axis =0 ),0 )/np .mean (O0OOO00OO000OOO0O ,axis =0 ).max ()#line:217
        O0OOOOO0O0000000O .append (O0OOO00OO000OOO0O )#line:218
        OO0OOOO0OOO0O0000 .append (OOO0O000O0OO0OO0O )#line:219
    return O0OOOOO0O0000000O ,np .array (OO0OOOO0OOO0O0000 ),np .mean (np .array (OO0OOOO0OOO0O0000 ),axis =0 )#line:220
def backOneHot (OO0O0000O00O0O000 ):#line:221
    return np .argmax (OO0O0000O00O0O000 ,axis =1 )#line:222
from sklearn .model_selection import train_test_split ,GridSearchCV #line:223
from sklearn .svm import SVC #line:224
def skGridSearchCv (O0O0O0OO0O0OOOO0O ,OO0O0O0OO00O0O00O ,O00OOOO0OO0OO0O00 ,OOO0OOOOOO00000OO ,OO00000O00OO00000 ):#line:225
    ""#line:226
    OOOO000OO0O00O000 ={"gamma":np .logspace (0 ,9 ,11 ,base =OO00000O00OO00000 ),"C":0.01 *np .logspace (0 ,9 ,11 ,base =OO00000O00OO00000 )}#line:227
    print ("Parameters:{}".format (OOOO000OO0O00O000 ))#line:228
    OOO0OOOO0OOOO0OOO =GridSearchCV (SVC (),OOOO000OO0O00O000 ,cv =5 ,n_jobs =10 )#line:229
    OOO0OOOO0OOOO0OOO .fit (O0O0O0OO0O0OOOO0O ,O00OOOO0OO0OO0O00 )#line:230
    print ("Test set score:{:.5f}".format (OOO0OOOO0OOOO0OOO .score (OO0O0O0OO00O0O00O ,OOO0OOOOOO00000OO )))#line:231
    print ("Best parameters:{}".format (OOO0OOOO0OOOO0OOO .best_params_ ))#line:232
    print ("Best score on train set:{:.5f}".format (OOO0OOOO0OOOO0OOO .best_score_ ))#line:233
    return OOO0OOOO0OOOO0OOO .score (OO0O0O0OO00O0O00O ,OOO0OOOOOO00000OO )#line:234
def alter_model (layer_name ='C4'):#line:235
    register_gradient ()#line:236
    OO0O00O00O00OO0O0 =creatModelTest ()#line:237
    OO0O00O00O00OO0O0 .load_weights ('weights.best.hdf5')#line:238
    O0OO000OOOOOOOOO0 =modify_backprop (OO0O00O00O00OO0O0 ,'GuidedBackProp')#line:239
    OOO0OO00OO0OOOO0O =compile_saliency_function (O0OO000OOOOOOOOO0 ,activation_layer =layer_name )#line:240
    return OOO0OO00OO0OOOO0O ,OO0O00O00O00OO0O0 #line:241
model =creatModelTest ()#line:242
model .load_weights ('weights.best.hdf5')#line:243
S ,Y ,T =get_all_result (m =9 ,name ='C3')#line:244
L =[]#line:245
for i in [5 ,10 ,15 ,20 ,25 ,30 ,35 ,40 ,45 ,50 ]:#line:246
     #line:247
    selectedtrainband ,selectedtestband ,selection ,selectedtrainlabel ,selectedtestlabel =bandSelectionAverage (train_data ,test_data ,train_label ,test_label ,Y,i,9 )#line:248
    acc =skGridSearchCv (selectedtrainband ,selectedtestband ,selectedtrainlabel ,selectedtestlabel ,3 )#line:249
    L .append (acc )#line:250
print ("[5,10,15,20,25,30,35,40,45,50] accuracy:",L )#line:251
