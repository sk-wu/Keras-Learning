#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:w_sk time:2019/7/7

import keras
from keras.layers import Input, BatchNormalization, Dense, Dropout, Flatten
from keras.models import Model
number_classes = 5


class FlowerNet(object):
    def __init__(self, input_width, input_height, channels_num):
        self.input_width = input_width
        self.input_height = input_height
        self.channels_num = channels_num

    def model(self):
        input = Input(shape=(self.input_width, self.input_height, self.channels_num))
        vgg_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=input)
        vgg_model.summary()
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
        x = layer_dict['block2_pool'].output
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(number_classes, activation='softmax')(x)
        new_model = Model(inputs=vgg_model.input, outputs=x)
        for layer in new_model.layers[:7]:
            layer.trainable = True
        new_model.summary()
        return new_model
