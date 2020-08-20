# 删除图片，保证图片和标签文件夹内容和名称匹配

import os

for file in os.listdir("./data/location/images"):
    file_name = os.path.join("./data/location/images", file)
    file_label = file_name.replace("images", "labels").replace("jpg", "json")
    print(file_name)
    print(file_label)
    if os.path.exists(file_label) is False:
        os.remove(file_name)
