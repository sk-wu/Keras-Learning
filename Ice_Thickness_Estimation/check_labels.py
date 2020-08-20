# 查看标注是否正确, 将标注结果在标注的原图显示出来

import os
import cv2

label_dir = "./data/location/labels"


def check_img():
    for label in os.listdir(label_dir):
        if not label.endswith("json"):
            continue
        label_path = os.path.join(label_dir, label)
        print(label_path)
        with open(label_path, "r") as f:
            data = f.read()
            data = eval(data)['region']
            print(data)

        loc = []
        for i in range(len(data)):
            loc.append(data[i]['x'])
        print(loc)

        img_file = label_path.replace("labels", "images").replace("json", "jpg")
        img = cv2.imread(img_file)
        sp = img.shape
        cv2.line(img, (int(loc[0]), 0), (int(loc[3]), sp[0]), (0, 0, 255), 1)
        cv2.line(img, (int(loc[1]), 0), (int(loc[2]), sp[0]), (0, 0, 255), 1)
        cv2.imshow("ttt", img)
        cv2.waitKey()


check_img()