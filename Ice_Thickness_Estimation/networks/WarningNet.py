# 分类网络

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential


class WarningNet(object):
    def __init__(self, width, height, channels_num):
        self.width = width
        self.height = height
        self.channels_num = channels_num

    def build_classification_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(self.height, self.width, self.channels_num),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax', name='output'))
        model.summary()
        return model


k = WarningNet(240, 27, 3)
k.build_classification_model()
