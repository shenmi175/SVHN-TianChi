import os
import json
import cv2
from albumentations import Compose, RandomBrightnessContrast, Rotate, Blur, RGBShift
import numpy as np
from tqdm import tqdm


# 图像增强配置
def get_augmentation(extra=False):
    return Compose([
        RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0 if extra else 0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0 if extra else 0.5),
        Rotate(limit=25, p=1.0 if extra else 0.5),
        Blur(blur_limit=3, p=1.0 if extra else 0.5)
    ])


# 加载和解析标注文件
def load_annotations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


# 图像增强应用函数
def apply_augmentations(image_folder, annotations, output_image_folder, output_json_folder):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_json_folder, exist_ok=True)
    augmented_annotations = {}

    # tqdm 包裹了外层循环
    for filename in tqdm(annotations.keys(), desc="Processing images"):
        attrs = annotations[filename]
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue  # 如果图片不存在或读取失败，跳过此图片
        labels = attrs['label']
        extra = any(label in [1, 2, 3] for label in labels)

        # 确定增强次数
        # 对部分类别额外增强
        num_augmentations = 2 if extra else 1

        for i in range(num_augmentations):
            augmented_image = get_augmentation(extra)(image=image)['image']
            new_filename = f"{filename.split('.')[0]}_A{i + 1}.png"
            cv2.imwrite(os.path.join(output_image_folder, new_filename), augmented_image)

            # 更新标注信息
            augmented_annotations[new_filename] = attrs

    # 保存增强后的标注文件
    with open(os.path.join(output_json_folder, 'augmented_annotations_B.json'), 'w') as file:
        json.dump(augmented_annotations, file)

if __name__ == "__main__":
    # 主流程
    annotations = load_annotations('json/mchar_train.json')
    apply_augmentations('dataset/images/train', annotations, 'New_Data/images', 'New_Data')
