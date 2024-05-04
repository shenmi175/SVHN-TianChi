# 可以参考说明文档将模型转为其它类型的格式，并添加全局nms

from ultralytics import YOLO

model = YOLO('yolo/train.pt')
model.export(format='onnx', dynamic=True)