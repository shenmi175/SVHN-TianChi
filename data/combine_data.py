from process_json import count_images
import shutil
import os
from tqdm import tqdm

def merge_folders(source_folders, target_folder, file_prefixes):
    """将多个源文件夹的文件复制到目标文件夹，为重复的文件添加前缀以区分来源，并显示进度条"""
    os.makedirs(target_folder, exist_ok=True)  # 确保目标文件夹存在

    # 计算所有源文件夹中的文件总数以初始化进度条
    total_files = sum(len(os.listdir(folder)) for folder in source_folders)
    progress_bar = tqdm(total=total_files, desc="Copying files")

    for folder, prefix in zip(source_folders, file_prefixes):
        for filename in os.listdir(folder):
            source_file = os.path.join(folder, filename)
            new_filename = f"{prefix}_{filename}"
            target_file = os.path.join(target_folder, new_filename)
            if not os.path.exists(target_file):  # 检查目标文件夹中是否已有该文件
                shutil.copy(source_file, target_file)
                progress_bar.update(1)  # 更新进度条
            else:
                print(f"File {new_filename} already exists in {target_folder}")

    progress_bar.close()


if __name__ == "__main__":
    # 源文件夹和目标文件夹
    image_source_folders = ['extra_data/train', 'extra_data/test', 'extra_data/extra']  # 图片源文件夹路径
    annotation_source_folders = ['extra_data/labels_train', 'extra_data/labels_test', 'extra_data/labels_extra']  # 标注源文件夹路径
    image_target_folder = 'images/train3'  # 目标图片文件夹路径
    annotation_target_folder = 'labels/train3'  # 目标标注文件夹路径

    # 定义文件前缀以区分不同源的文件
    image_prefixes = ['train', 'test', 'extra']
    annotation_prefixes = ['train', 'test', 'extra']

    # 合并图片文件夹
    merge_folders(image_source_folders, image_target_folder, image_prefixes)
    # 合并标注文件夹
    merge_folders(annotation_source_folders, annotation_target_folder, annotation_prefixes)
    # 打印图片数量
    image_count = count_images(image_target_folder)
    print(f"图片数量：{image_count}")
