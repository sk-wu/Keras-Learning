# 训练定位模型

import keras
from networks import IceNet
from keras.models import load_model
import numpy as np
import cv2
import os
import random

image_extension = "jpg"
label_extension = "json"
img_dir = './data/location/train_images'
model_name = './models/loc_model/loc_model_'
input_width = 240
input_height = 27
channels_num = 3


def preprocess(src_img):
    norm_img = src_img / 255
    resize_img = cv2.resize(norm_img, (input_width, input_height))
    reshape_img = np.resize(resize_img, (input_height, input_width, channels_num))
    return reshape_img


# def loadata(img_dir, ice_net):
#     img_data = []
#     label_data = []
#     img_path = os.listdir(img_dir)
#
#     for file in img_path:
#         img = cv2.imread(os.path.join(img_dir, file))
#         input_img = preprocess(img)
#         img_data.append(input_img)
#         batch_img_path = os.path.join(img_dir, file).replace(image_extension, label_extension).replace("images", "labels")
#         cur_label = []
#         with open(batch_img_path, "r") as fr:
#             positions = eval(fr.read())
#             for index, element in enumerate(positions):
#                 cur_label.append(element / input_width)
#             label_data.append(cur_label)
#     # print(cur_label)
#     return np.array(img_data), np.array(label_data)

def loadata(img_dir):
    img_data = []
    label_data = []
    img_path = os.listdir(img_dir)

    for file in img_path:
        img = cv2.imread(os.path.join(img_dir, file))
        # print(img.shape) # 依次输出图片的高度、宽度、和通道数
        input_img = preprocess(img)
        # print(input_img.shape)
        img_data.append(input_img)
        batch_img_path = os.path.join(img_dir, file).replace(image_extension, label_extension).replace("images", "labels")
        cur_label = []
        with open(batch_img_path, "r") as fr:
            positions = eval(fr.read())
            for index, element in enumerate(positions):
                cur_label.append(element)
            print(cur_label)
            label_data.append(cur_label)
            # print(cur_label)

    # 打乱数据
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(img_data)
    random.seed(randnum)
    random.shuffle(label_data)

    return np.array(img_data), np.array(label_data)


def train():
    # 构建新的模型
    # loc_net = IceNet.IceNet(input_width, input_height, channels_num)
    # model = loc_net.build_simple_model()
    # # print(model.summary)

    # 载入预训练模型
    model = load_model('./models/loc_model/pretrained_loc_model_200.h5')
    print(model.summary)

    print("Start training...")
    train_data, train_label = loadata(img_dir)
    print(len(train_data))
    # print(train_data)
    # print(train_label)
    print("Successfully load {} pictures and labels...".format(len(train_data)))

    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam, metrics=['mse'])
    for i in range(300):
        print("---------------------------------第{}次迭代---------------------------------".format(i+1))
        model.fit(x=train_data, y=train_label, validation_split=0.1, epochs=5, batch_size=200, verbose=1)
        if i > 0 and i % 5 == 0:
            model.save(model_name + str(i)+".h5")
            print("Successfully save model:", model_name + str(i) + ".h5")


if __name__ == '__main__':
    train()
