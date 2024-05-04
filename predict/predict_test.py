import os
import csv
from ultralytics import YOLO
from pathlib import Path


class StreetSceneDetector:
    def __init__(self, model_path, input_folder, output_file):
        self.model = YOLO(model_path)
        self.input_folder = input_folder
        self.output_file = output_file

    def detect_and_save(self):
        results = self.model.predict(self.input_folder,
                                     device=0,  # 使用GPU
                                     save=False,  # 不保存预测图片
                                     show_labels=False,  # 不显示标签
                                     imgsz=320,  # 输入图片尺寸
                                     batch=256,  # 使用批量处理
                                     stream=False,  # 返回所有结果

                                     # 0.9287-->0.9354
                                     iou=0.5,
                                     conf=0.001,
                                     agnostic_nms=True,
                                     )

        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['file_name', 'file_code'])  # 写入CSV列名

            for result in results:
                img_path = result.path  # 使用path属性获取图像路径
                file_name = os.path.basename(img_path)
                if hasattr(result, 'boxes'):
                    file_code = self.extract_codes(result.boxes)  # 整数或None
                    if file_code is not None:
                        writer.writerow([file_name, file_code])  # 直接写入整数值
                    else:
                        writer.writerow([file_name, ''])
                else:
                    writer.writerow([file_name, ''])

    @staticmethod
    def extract_codes(boxes):
        # 先将boxes对象按照边界框的左上角x坐标（xyxy数组的第一个元素）排序
        # 确保在访问前将Tensor切片为单个标量，并转移到CPU
        sorted_boxes = sorted(boxes, key=lambda box: box.xyxy[:, 0].cpu().numpy())
        # 提取排序后的类别代码
        codes = [int(box.cls) for box in sorted_boxes if box.conf > 0.15]  # 置信度大于0.15的预测
        if codes:  # 确保列表不为空
            combined_codes = ''.join(map(str, codes))  # 直接合并数字，不进行反转
            return int(combined_codes)  # 将合并后的字符串转换为整数
        else:
            return None  # 如果没有有效代码，则返回None

if __name__ == "__main__":
    print('开始')
    detector = StreetSceneDetector('../train/runs/detect/train/weights/best.pt',
                                   '../data/mchar_test_a',
                                   'results/output-pass.csv')
    detector.detect_and_save()
