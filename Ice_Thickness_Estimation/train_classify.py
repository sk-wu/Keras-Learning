# 训练分类器

import keras
from networks import WarningNet
from keras.models import load_model


num_classes = 2
train_data_dir = './data/classification/train'
validation_data_dir = './data/classification/validation'
model_name = './models/classify_model/classify_model_'
input_width = 240
input_height = 27
channels_num = 3


def train():
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(input_height, input_width),  # 目标大小（target_size）：整数的元组（高度、宽度）
                                                        batch_size=32,
                                                        shuffle=True,
                                                        class_mode='categorical')

    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(input_height, input_width),
                                                                  batch_size=24,
                                                                  class_mode='categorical')

    # 新建模型
    # loc_object = WarningNet.WarningNet(input_width, input_height, channels_num)
    # loc_model = loc_object.build_classification_model()

    # 载入预训练模型
    loc_model = load_model('./models/classify_model/pretrained_classify_model.h5')
    print(loc_model.summary)



    # 编译与训练
    adam = keras.optimizers.Adam(lr=0.001)
    loc_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    for i in range(50):
        loc_model.fit_generator(train_generator, epochs=1, steps_per_epoch=80, validation_data=validation_generator,
                                validation_steps=80)
        if i > 0 and i % 2 == 0:
            loc_model.save(model_name + str(i) + ".h5")


train()
