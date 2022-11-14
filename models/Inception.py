# https://github.com/maelfabien/Machine_Learning_Tutorials/blob/master/4-DeepLearning/Inception/model.png

import numpy as np
import keras 
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers import Input

from keras.models import Model




def Inception(input_shape, num_classes):
    input_img = Input(shape=input_shape)

    ### 1st layer
    layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
    layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)

    ### 2nd layer
    layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
    layer_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer_2)

    ### 3rd layer
    layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    layer_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_3)

    ### Concatenate
    mid_1 = keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)

    flat_1 = Flatten()(mid_1)

    dense_1 = Dense(900, activation='relu')(flat_1)
    dense_2 = Dense(600, activation='relu')(dense_1)
    dense_3 = Dense(150, activation='relu')(dense_2)

    output = Dense(num_classes, activation='softmax')(dense_3)

    return Model([input_img], output)

