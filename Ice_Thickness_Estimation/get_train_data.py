# 获取训练需要的图片和标签，将坐标转换成比例(即转换成训练需要的格式)

import os
import cv2
input_width = 240
input_height = 27
label_dir = './data/location/labels'
img_dir = './data/location/images'
save_label_dir = './data/location/train_labels'
save_img_dir = "./data/location/train_images"
ratio = 1920 // input_width


def get_train_images():
    for file in os.listdir(img_dir):
        f = os.path.join(img_dir, file)
        img = cv2.imread(f)
        new_img = cv2.resize(img, (input_width, input_height))
        cv2.imwrite(os.path.join(save_img_dir, file), new_img)


def get_train_labels():
    for label in os.listdir(label_dir):
        if not label.endswith("json"):
            continue
        label_path = os.path.join(label_dir, label)
        save_label_path = os.path.join(save_label_dir, label)
        print(save_label_path)
        with open(label_path, "r") as f:
            data = f.read()
            data = eval(data)['region']
            print(data)

        loc = []
        for i in range(len(data)):
            loc.append(data[i]['x']/ratio)
        print(loc)

        with open(save_label_path, "w") as fw:
            fw.write(str(loc))


if __name__ == '__main__':
    get_train_images()
    get_train_labels()
