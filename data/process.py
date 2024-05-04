# 将标注文件yolo格式

import json
from PIL import Image
import os
from tqdm import tqdm

# 定义存放图片的文件夹路径
image_folder = '../dataset/data/images'

# 创建标注文件存放的文件夹
annotations_dir = '../dataset/data/labels'

os.makedirs(annotations_dir, exist_ok=True)

# 假设你的 JSON 数据存储在 'annotations.json' 文件中
with open('json/train.json', 'r') as file:
    data = json.load(file)

# 使用 tqdm 包装字典的 items() 迭代器以显示进度条
for img_id, attrs in tqdm(data.items(), desc="Processing images"):
    img_path = os.path.join(image_folder, img_id)

    try:
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        annotation_filename = os.path.join(annotations_dir, f"{os.path.splitext(os.path.basename(img_id))[0]}.txt")

        with open(annotation_filename, 'w') as out_file:
            for i in range(len(attrs['label'])):
                top = attrs['top'][i]
                left = attrs['left'][i]
                width = attrs['width'][i]
                height = attrs['height'][i]

                x_center = (left + width / 2) / img_width
                y_center = (top + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height

                out_file.write(f"{attrs['label'][i]} {x_center} {y_center} {norm_width} {norm_height}\n")
    except IOError:
        print(f"Error opening or reading image file: {img_id}")
