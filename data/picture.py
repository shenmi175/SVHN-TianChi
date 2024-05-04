import os
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 设置图片和标注文件的文件夹路径
# image_folder = 'train'
# annotations_dir = 'train'

image_folder = '../dataset/data/train'
annotations_dir = '../dataset/data/train_labels'

# 获取所有图片文件
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 从图片列表中随机选择一张图片
selected_image = random.choice(image_files)

# 构建标注文件的路径
annotation_file = os.path.join(annotations_dir, os.path.splitext(selected_image)[0] + '.txt')

# 打开图片文件
img_path = os.path.join(image_folder, selected_image)
image = Image.open(img_path)
img_width, img_height = image.size

# 准备绘图
draw = ImageDraw.Draw(image)

# 读取标注信息并绘制边界框
try:
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            obj_class, x_center, y_center, width, height = map(float, parts)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            left = x_center - width / 2
            top = y_center - height / 2
            right = x_center + width / 2
            bottom = y_center + height / 2

            # 绘制矩形框
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

    # 显示图片
    plt.imshow(image)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
except IOError:
    print(f"Error opening or reading annotation file: {annotation_file}")
