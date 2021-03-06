# 定位网络

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, AveragePooling2D, concatenate, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class IceNet():
    def __init__(self, width, height, channels_num):
        super(IceNet, self).__init__()
        self.width = width
        self.height = height
        self.channels_num = channels_num

    def build_lpr_model(self):
        input = Input(shape=(self.height, self.width, self.channels_num))
        x1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input)
        x21 = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x1)
        x22 = Conv2D(filters=32, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu')(input)
        x2_concat = concatenate([x21, x22])
        dr = Dropout(0.5)(x2_concat)
        fl = Flatten()(dr)
        output = Dense(128, activation='relu')(fl)
        output = Dense(4)(output)
        model = Model(input, output)
        # model.summary()
        return model

    def build_loc_model(self):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(self.height, self.width, self.channels_num), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4))
        model.summary()
        return model

    def build_squeezenet_model(self):
        input = Input(shape=(self.height, self.width, self.channels_num))
        network = Conv2D(64, (3, 3), strides=(2, 2), padding="valid", activation='relu')(input)

        k1 = Conv2D(16, (1, 1), padding="valid", activation='relu')(network)
        input_branch_1 = Conv2D(64, (1, 1), padding="valid", activation='relu')(k1)
        input_branch_2 = Conv2D(64, (3, 3), padding="same", activation='relu')(k1)
        network = concatenate([input_branch_1, input_branch_2])
        network = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(network)

        k2 = Conv2D(16, (1, 1), padding="valid", activation='relu')(network)
        input_branch_1 = Conv2D(64, (1, 1), padding="valid", activation='relu')(k2)
        input_branch_2 = Conv2D(64, (3, 3), padding="same", activation='relu')(k2)
        network = concatenate([input_branch_1, input_branch_2])

        k3 = Conv2D(32, (1, 1), padding="valid", activation='relu')(network)
        input_branch_1 = Conv2D(128, (1, 1), padding="valid", activation='relu')(k3)
        input_branch_2 = Conv2D(128, (3, 3), padding="same", activation='relu')(k3)
        network = concatenate([input_branch_1, input_branch_2])

        k4 = Conv2D(32, (1, 1), padding="valid", activation='relu')(network)
        input_branch_1 = Conv2D(128, (1, 1), padding="valid", activation='relu')(k4)
        input_branch_2 = Conv2D(128, (3, 3), padding="same", activation='relu')(k4)
        network = concatenate([input_branch_1, input_branch_2])

        network = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(network)

        # Remove layers like Dropout and BatchNormalization, they are only needed in training
        network = Dropout(0.5)(network)
        network = Conv2D(1000, kernel_size=(1, 1), padding="valid", name="last_conv", activation="relu")(network)
        network = Flatten()(network)
        network = Dense(128, activation='relu')(network)
        output = Dense(4, name="last")(network)
        model = Model(inputs=input, outputs=output)
        model.summary()
        return model

    def build_resnet50(self):
        input = Input(shape=(self.height, self.width, self.channels_num))
        res_model = keras.applications.ResNet50(include_top=True, weights='imagenet', input_tensor=input)
        # res_model.summary()
        layer_dict = dict([(layer.name, layer) for layer in res_model.layers])
        # print(layer_dict)
        x = layer_dict['avg_pool'].output
        output = Dense(4, name="last")(x)
        model = Model(inputs=input, outputs=output)
        model.summary()
        for layer in model.layers[:100]:
            layer.trainable = False
        return model


    def build_simple_model(self):
        input = Input(shape=(self.height, self.width, self.channels_num))
        res_model = keras.applications.ResNet50(include_top=True, weights='imagenet', input_tensor=input)
        # res_model.summary()
        layer_dict = dict([(layer.name, layer) for layer in res_model.layers])
        # print(layer_dict)
        x = layer_dict['activation_11'].output
        x = GlobalAveragePooling2D()(x)
        output = Dense(4, name="last")(x)
        model = Model(inputs=input, outputs=output)
        model.summary()
        for layer in model.layers[:]:
            layer.trainable = True
        return model



# obj = IceNet(240, 27, 3)
# obj.buld_resnet50()
# input = Input(shape=(240, 27, 3))
# res_model = keras.applications.ResNet50(include_top=True, weights='imagenet', input_tensor=input)
# # res_model = keras.applications.ResNet50(include_top=True, weights='imagenet')
# res_model.summary()

# input = Input(shape=(240, 240, 3))
# vgg_model = keras.applications.VGG16(include_top=True, weights='imagenet', input_tensor=input)
# vgg_model.sumary()