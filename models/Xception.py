import numpy as np

import tensorflow as tf
import tensorflow.keras

from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D





def Xception(input_shape, num_classes):
    def entry_flow(inputs) :
        
        x = Conv2D(32, 3, strides = 2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(64,3,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        previous_block_activation = x
        
        for size in [128, 256, 728] :
        
            x = Activation('relu')(x)
            x = SeparableConv2D(size, 3, padding='same')(x)
            x = BatchNormalization()(x)
        
            x = Activation('relu')(x)
            x = SeparableConv2D(size, 3, padding='same')(x)
            x = BatchNormalization()(x)
            
            x = MaxPooling2D(3, strides=2, padding='same')(x)
            
            residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)
            
            x = tensorflow.keras.layers.Add()([x, residual])
            previous_block_activation = x
        
        return x

    def middle_flow(x, num_blocks=8) :
        
        previous_block_activation = x
        
        for _ in range(num_blocks) :
        
            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)
        
            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)
            
            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)
            
            x = tensorflow.keras.layers.Add()([x, previous_block_activation])
            previous_block_activation = x
        
        return x


    def exit_flow(x) :
        
        previous_block_activation = x
        
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = Activation('relu')(x)
        x = SeparableConv2D(1024, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = MaxPooling2D(3, strides=2, padding='same')(x)
        
        residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
        x = tensorflow.keras.layers.Add()([x, residual])
        
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = Activation('relu')(x)
        x = SeparableConv2D(1024, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='softmax')(x)
        
        return x


    inputs = Input(shape=input_shape)
    outputs = exit_flow(middle_flow(entry_flow(inputs)))
    return Model(inputs, outputs)


    # plot_model(xception, to_file='model.png', show_shapes=True, show_layer_names=True)