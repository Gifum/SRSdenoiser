# SRSdenoiser: NN model for SRS denoising and baseline remotion
### See the main text and SI of the paper for additional details


### Requirements
import numpy as np
import os
import array
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import norm
from tensorflow.keras.callbacks import History
history = History()



def Multi_parallelKernel_CNN_modified3(input_size, nlayers=6, k_size=[80,20,5],channels=[32,64,128],bn=False, nDense=0,nConvFinal=0):
    ''' 
    Instantiate the model for the SRSdenoiser neural network.
    Inputs:
        input_size is the size of the input array
        nlayers the number of parallel convolutional branches. It must be an integer
        k_size is an array containing the sizes of the kernels on each convolutional branch
        channels is an array containing the number of channels on each convolutional branch
        bn enables/remove batch normalization. It can be set to True or False
        nDense is the number of fully connected layers before the parallel convolutional branches
        nConvFinal is the number of convolutional layers with decreasing kernel sizes after the concatenation of the parallel convolutional branches 
    '''    
    
    nBranch=np.shape(k_size)[0]
    #Convert input list element in int
    channels = array.array('i',channels)
    k_size = array.array('i',k_size)
    

    
    inputs = tf.keras.Input(shape=input_size)

    a= inputs
    
    for nd in range(nDense):
        a = layers.Dense(64,name='DenseLayer_'+str(nd),kernel_initializer=tf.keras.initializers.HeUniform())(a)
        a=layers.Activation('relu')(a) 

    # Concatenate multiple parallel convolutional branches with different kernel sizes
    
    for nb in range(nBranch):
        x = layers.Conv1D(channels[nb], k_size[nb], padding="same",name='Conv'+str(nb)+'Layer0')(a)
        x = layers.Activation('relu',name='Act'+str(nb)+'Layer0')(x)

        for i in range(nlayers):
            x = layers.Conv1D(channels[nb], k_size[nb], padding='same',name='Conv'+str(nb)+'Layer'+str(i+1))(x)
            if bn:
                x = layers.BatchNormalization(axis=-1, epsilon=1e-3)(x)
            x = layers.Activation('relu',name='Act'+str(nb)+'Layer'+str(i+1))(x)
        
        if nb>0:
            f=layers.Concatenate(axis=-1)([f,x])
        else:
            f=x
    
    for nf in range(nConvFinal):

        f = layers.Conv1D(f.shape[-1]/2, 1, padding='same',name='ConvFinalLayer_'+str(nf))(f)
        f=layers.Activation('relu')(f) 

    f = layers.Conv1D(1, 1, padding='same')(f)
    f = layers.Subtract()([inputs, f])   # input - noise (residual layer)

    model = keras.Model(inputs=inputs, outputs=f)

    model.save_weights('init_weights.h5')
    
    return model