import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, Conv3D, BatchNormalization, Activation,Concatenate, Conv3DTranspose
from keras.layers import AveragePooling2D, Input, Flatten,MaxPooling3D,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10

from keras.utils import plot_model
from keras.optimizers import SGD
import numpy as np
import os
from keras.losses import BinaryCrossentropy

from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from keras import objectives
from sklearn.model_selection import train_test_split

from keras import objectives, backend as K
import keras
import tensorflow as tf

from tensorflow.python.client import device_lib
from Inputs import *
from tensorflow.keras.utils import multi_gpu_model

#gpus = tf.config.experimental.list_physical_devices('GPU')
    
#tf.config.experimental.set_visible_devices([gpus[5],gpus[6],gpus[7],gpus[8],gpus[9],], device_type='GPU')
                                                                                   
tf.debugging.set_log_device_placement(True)




def feature_extractor(inputs,regularizer = 1e-6,tempConvSize = 3):

    x = Conv3D(64, (tempConvSize, 3, 3), kernel_regularizer=l2(regularizer), 
           padding='same', name='current_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                 padding='valid', name='current_pool1')(x)

    x = Conv3D(128, (tempConvSize, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                 padding='valid', name='current_pool2')(x)

    x = Conv3D(256, (tempConvSize, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv3a')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(256, (tempConvSize, 3, 3), kernel_regularizer=l2(regularizer),
          padding='same', name='current_conv3b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                 padding='valid', name='current_pool3')(x)

    x = Conv3D(512, (tempConvSize, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv4a')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (tempConvSize, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv4b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                 padding='valid', name='current_pool4')(x)

    x = Conv3D(512, (tempConvSize, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv5a')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (tempConvSize, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv5b')(x)
    x = BatchNormalization()(x)
    exFeat = Activation('relu')(x)

    return exFeat




def coarse_predictor(exFeat,regularizer = 1e-6):
    x = Conv3D(512, (3, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv6a')(exFeat)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(512, (1, 1, 1), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv6b')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(1, (1, 1, 1), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv6c')(x)

    x = BatchNormalization()(x)
    cap = Activation('linear')(x)


    return cap





def Upsampling(exFeat, cap,regularizer):
    x = Concatenate()([exFeat, cap])
    x = Conv3DTranspose(512, (1, 4, 4), strides=(1, 4, 4), padding='valid', name='current_deconv_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv7a')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv7b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3DTranspose(512, (1, 4, 4), strides=(1, 4, 4), padding='valid', name='current_deconv_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv8a')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv8b')(x)
    x = BatchNormalization()(x)
    up = Activation('relu')(x)

    return up




def fine_predictor(up,regularizer):
    x = Conv3D(512, (1, 1, 1), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv11a')(up)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(1, (1, 1, 1), kernel_regularizer=l2(regularizer),
           padding='same', name='current_conv11b')(x)
    x = BatchNormalization()(x)

    fgp = Activation('linear')(x)


    return fgp


def coarse_model(input_shape = (16, 64,64, 3),regularizer = 1e-6,tempConvSize = 3):
    x_inp = Input(input_shape)
    x = feature_extractor(x_inp,regularizer = regularizer,tempConvSize = tempConvSize)
    x = coarse_predictor(x,regularizer = regularizer)
    return Model(inputs = x_inp, outputs = x, name='Coarse_Model')




def fine_model(input_shape = (16, 64,64, 3),regularizer = 1e-6,tempConvSize = 3):
    x_inp = Input(input_shape)
    feat = feature_extractor(x_inp,regularizer = regularizer,tempConvSize = tempConvSize)
    x = coarse_predictor(feat,regularizer = regularizer)
    up = Upsampling(feat,x,regularizer = regularizer)
    fine = fine_predictor(up,regularizer = regularizer)

    return Model(inputs = x_inp, outputs = fine, name='Fine_Model')


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


   
model1 = coarse_model()
model2 = fine_model()
#print(model1.summary())
#print(model2.summary())
#tf.compat.v1.Session().close()

#print('end of script')

def custom_loss(y_true,y_pred):
    
    y_true_flat = tf.reshape(y_true,(K.shape(y_pred)[0]*K.shape(y_pred)[1],K.shape(y_pred)[2]*K.shape(y_pred)[3]))
    y_pred_flat = tf.reshape(y_pred,(K.shape(y_pred)[0]*K.shape(y_pred)[1],K.shape(y_pred)[2]*K.shape(y_pred)[3]))
    #print(y_true_flat.shape,y_pred_flat.shape)
    #y_pred_flat = tf.nn.softmax(y_pred_flat,axis=-1)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  
    return cce(y_true_flat,y_pred_flat)


def KL_loss(y_true,y_pred):
    y_true_flat = tf.reshape(y_true,(K.shape(y_true)[0]*K.shape(y_true)[1],K.shape(y_true)[2]*K.shape(y_true)[3]))
    y_pred_flat = tf.reshape(y_pred,(K.shape(y_pred)[0]*K.shape(y_pred)[1],K.shape(y_pred)[2]*K.shape(y_pred)[3]))
    y_pred_flat = tf.nn.softmax(y_pred_flat,axis=-1)
    #print(K.shape(y_pred_flat),K.shape(y_true_flat))
    KL = tf.keras.losses.KLDivergence()  
    return KL(y_true_flat,y_pred_flat)
  

def dist(y_true,y_pred):
      y_true = tf.reshape(y_true,(K.shape(y_true)[0]*K.shape(y_true)[1],K.shape(y_true)[2],K.shape(y_true)[3]))
      y_pred = tf.reshape(y_pred,(K.shape(y_pred)[0]*K.shape(y_pred)[1],K.shape(y_pred)[2],K.shape(y_pred)[3]))
      x0_true = K.cast(K.argmax(K.max(y_true,axis=-2)),dtype='float64')
      y0_true = K.cast(K.argmax(K.max(y_true,axis=-1)),dtype='float64')
      x0_pred = K.cast(K.argmax(K.max(y_pred,axis=-2)),dtype='float64')
      y0_pred = K.cast(K.argmax(K.max(y_pred,axis=-1)),dtype='float64') 
      return K.mean(((K.square((x0_true-x0_pred))+K.square((y0_true-y0_pred))))**0.5)


    
    
def generate_plots(df):
    history = pd.read_csv(df)
    plt.plot(history.iloc[:,1],label='val loss')
    plt.plot(history.iloc[:,3],label='train loss')
    plt.legend()
    plt.savefig('loss_plot'+(df)+'.png')
    plt.close()
    plt.plot(history.iloc[:,2],label='val deviation')
    plt.plot(history.iloc[:,4],label='train deviation')
    plt.legend()
    plt.show()
    plt.savefig('acc_plot'+df+'.png')
    plt.close()

if __name__=='__main__':
    
    
    
    
    
# Open a strategy scope.
    
    #print(gpus)
    print('done')
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:5','/gpu:6'])
    #with mirrored_strategy.scope():
    gpus = 5
    if True:
        #model = multi_gpu_model(model, gpus=gpus)
    
        train_X,frames_array = get_frames()
        train_Y,heatmap = get_grids()
        #train_X, X_test, heatmap, y_test = train_test_split(train_X, heatmap, test_size=0.20, random_state=42)
        opt = SGD(lr=0.0005, momentum=0.9, decay=1e-2)
        
        
        model1.compile(loss= custom_loss, optimizer=opt)
        history = model1.fit(train_X,train_Y,batch_size=32,epochs=200)#,validation_data=(X_test, y_test))
        #hist_df = pd.DataFrame(history.history) 
        #hist_csv_file = 'history_valfine_2.csv'
        #with open(hist_csv_file, mode='w') as f:
        #    hist_df.to_csv(f)
        
        #generate_plots(hist_csv_file)


   
    
        #model1 = coarse_model()
    #model2 = fine_model()
        #batch_size=16
    
    
    
    #np.save('heatmap',heatmap)
    #train_X,train_Y = shuffle_in_unison(np.array(train_X),np.array(train_Y))
    
    
    
    
    
    #print(heatmap.shape)
  
       
    
    
       
    ##model2.save_weights("fine_mod(200epch).h5")
    ##hist_df = pd.DataFrame(history.history) 
    ##hist_csv_file = 'history_valfine_2.csv'
    ##with open(hist_csv_file, mode='w') as f:
     ##   hist_df.to_csv(f)
        
    ##generate_plots(hist_csv_file)
    #train_X,frames_array = get_frames()
    #train_Y,heatmap = get_grids()
    #train_X, X_test, train_Y, y_test = train_test_split(train_X, train_Y, test_size=0.20, random_state=42)
    #model1.compile(loss = custom_loss,optimizer = opt,metrics = ['accuracy'])
    
    #history1 = model1.fit(train_X,train_Y,batch_size=16,epochs=300,validation_data=(X_test, y_test))
   
    #model1.save_weights("coarse_mod(300epch).h5")
    #hist_df = pd.DataFrame(history1.history) 
    #hist_csv_file1 = 'history_valcoarse_2.csv'
    #with open(hist_csv_file1, mode='w') as f:
    #    hist_df.to_csv(f)
    
    #generate_plots(hist_csv_file1)


    #model1.save_weights("model_coarse(10).h5")
