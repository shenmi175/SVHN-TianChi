import os
import json
import cv2


def count_images(image_folder):
    """打印指定文件夹下的图片数量"""
    image_files = [file for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"There are {len(image_files)} images in the folder '{image_folder}'.")
    return len(image_files)


def convert_annotations_to_yolo(json_file, yolo_txt_folder, image_folder):
    """解析JSON文件并转换为YOLO格式"""
    os.makedirs(yolo_txt_folder, exist_ok=True)
    with open(json_file, 'r') as file:
        annotations = json.load(file)

    for filename, attrs in annotations.items():
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width, _ = image.shape
        yolo_data = []
        for label, left, top, w, h in zip(attrs['label'], attrs['left'], attrs['top'], attrs['width'], attrs['height']):
            # 归一化坐标和尺寸
            x_center = (left + w / 2) / width
            y_center = (top + h / 2) / height
            norm_width = w / width
            norm_height = h / height
            yolo_data.append(f"{label} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        # 保存到对应的txt文件
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(yolo_txt_folder, txt_filename), 'w') as yolo_file:
            yolo_file.write('\n'.join(yolo_data))



if __name__ == "__main__":
    # 主流程
    image_folder = 'images'  # 图片文件夹路径
    json_file = 'augmented_annotations_B.json'  # 增强后的JSON文件路径
    yolo_txt_folder = 'labels'  # YOLO格式的标注文件存放路径


    # 打印图片数量
    count_images(image_folder)
    # 转为yolo格式
    convert_annotations_to_yolo(json_file, yolo_txt_folder, image_folder)