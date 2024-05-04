import os
import shutil
import csv
from ultralytics import YOLO
from collections import defaultdict


class ValidationDetector:
    def __init__(self, model_path, input_folder, label_folder, errors_folder, errors_label_folder, output_metrics_file):
        self.model = YOLO(model_path)
        self.input_folder = input_folder
        self.label_folder = label_folder
        self.errors_folder = errors_folder
        self.errors_label_folder = errors_label_folder
        self.output_metrics_file = output_metrics_file
        # 创建存储错误的文件夹和标签文件夹
        os.makedirs(self.errors_folder, exist_ok=True)
        os.makedirs(self.errors_label_folder, exist_ok=True)

    def validate_and_save_errors(self):
        results = self.model.predict(self.input_folder,
                                     device='cuda:0',
                                     save=False,
                                     show_labels=True,
                                     imgsz=320,
                                     conf=0.15,
                                     iou=0.5,
                                     batch=256,
                                     stream=False)

        metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0})
        global_total = 0
        global_correct = 0

        for result in results:
            img_id = os.path.splitext(os.path.basename(result.path))[0]
            pred_boxes = result.boxes
            pred_labels = self.extract_codes(pred_boxes) or ''  # 如果返回 None，则使用空字符串
            true_labels = self.load_labels(img_id)

            if pred_labels != true_labels:
                # 复制图像文件到错误文件夹
                shutil.copy(result.path, os.path.join(self.errors_folder, os.path.basename(result.path)))
                # 写入错误标签到单独的.txt文件中
                error_label_path = os.path.join(self.errors_label_folder, f"{img_id}.txt")
                with open(error_label_path, 'w') as file:
                    file.write(f"Predicted: {pred_labels}\n")
                    file.write(f"True: {true_labels}\n")

            # 更新指标
            pred_set = set(pred_labels)  # 确保 pred_labels 不是 None
            true_set = set(true_labels)
            for cls in true_set:
                if cls in pred_set:
                    metrics[cls]['tp'] += 1
                    global_correct += 1
                else:
                    metrics[cls]['fn'] += 1
            for cls in pred_set:
                if cls not in true_set:
                    metrics[cls]['fp'] += 1
            for cls in true_set:
                metrics[cls]['total'] += 1
                global_total += 1

        # 保存指标到CSV文件
        with open(self.output_metrics_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['class', 'precision', 'recall', 'accuracy'])
            for cls, metric in metrics.items():
                precision = metric['tp'] / (metric['tp'] + metric['fp']) if (metric['tp'] + metric['fp']) > 0 else 0
                recall = metric['tp'] / (metric['tp'] + metric['fn']) if (metric['tp'] + metric['fn']) > 0 else 0
                accuracy = metric['tp'] / metric['total'] if metric['total'] > 0 else 0
                writer.writerow([cls, precision, recall, accuracy])
            # 写入整体指标
            overall_precision = global_correct / global_total if global_total > 0 else 0
            overall_recall = global_correct / global_total if global_total > 0 else 0
            overall_accuracy = global_correct / global_total if global_total > 0 else 0
            writer.writerow(['Overall', overall_precision, overall_recall, overall_accuracy])

    @staticmethod
    def extract_codes(boxes):
        # 先将boxes对象按照边界框的左上角x坐标（xyxy数组的第一个元素）排序
        sorted_boxes = sorted(boxes, key=lambda box: box.xyxy[0, 0].cpu().numpy())
        # 提取排序后的类别代码
        codes = [str(int(box.cls)) for box in sorted_boxes if box.conf > 0.15]  # 置信度大于0.15的预测
        return ''.join(codes) if codes else None  # 直接合并数字，返回字符串形式的代码

    def load_labels(self, img_id):
        label_path = os.path.join(self.label_folder, f"{img_id}.txt")
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 0:
                    labels.append(parts[0])  # 假设标签文件中直接存储的是字符序列
        return ''.join(labels)  # 返回字符串形式的标签

if __name__ == "__main__":
    
    detector = ValidationDetector('../train/runs/detect/train/weights/best.pt',
                                  '../data/dataset/images/val',
                                  '../data/dataset/labels/val',
                                  'error_val/images',
                                  'error_val/labels',
                                  'error_val/error_2/validation_metrics.csv')
    detector.validate_and_save_errors()


