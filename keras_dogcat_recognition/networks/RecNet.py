#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:w_sk time:2019/4/2


import keras
from keras.layers import Input, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

num_classes = 2


class RecNet(object):
    def __init__(self, input_width, input_height, channels_num):
        self.input_width = input_width
        self.input_height = input_height
        self.channels_num = channels_num

    def model(self):
        input = Input(shape=(self.input_width, self.input_height, self.channels_num))
        vgg_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=input)

        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
        x = layer_dict['block5_conv3'].output
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(num_classes, activation="softmax")(x)

        new_model = Model(inputs=vgg_model.input, outputs=x)
        trainable = False
        for layer in new_model.layers:
            if layer.name == 'block5_conv1':
                trainable = True
            layer.trainable = trainable
        new_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-5),
                          metrics=['accuracy'])
        new_model.summary()
        return new_model

