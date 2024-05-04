# 导出模式的主要特点
以下是一些突出的功能：

- **一键导出**：简单的命令，用于导出到不同格式。
- **批量导出**：支持批量推理的模型导出。
- **优化推理**：导出的模型针对更快的推理时间进行了优化。
- **教程视频**：深入的指南和教程，确保顺畅的导出体验。

## 提示

- 导出到ONNX或OpenVINO，可实现高达3倍的CPU加速。
- 导出到TensorRT，可实现高达5倍的GPU加速。

## 使用示例
将YOLOv8n模型导出为ONNX或TensorRT等不同格式。下面的参数部分有完整的导出参数列表。

### 示例-Python
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # 加载官方模型
model = YOLO('path/to/best.pt')  # 加载自定义训练模型

# 导出模型
model.export(format='onnx')

```

# 参数
此表格详细介绍了将YOLO模型导出到不同格式的配置和选项。这些设置对于优化导出模型的性能、大小和在各种平台及环境中的兼容性至关重要。正确的配置确保模型可以在预定应用中以最佳效率部署。

| 参数        | 类型           | 默认值            | 描述                                                                                                 |
|-------------|----------------|------------------|-----------------------------------------------------------------------------------------------------|
| format      | str            | 'torchscript'    | 导出模型的目标格式，如'onnx'、'torchscript'、'tensorflow'等，定义了与各种部署环境的兼容性。          |
| imgsz       | int or tuple   | 640              | 模型输入的期望图像大小。可以是整数（对于正方形图像）或元组（高度，宽度）表示特定尺寸。               |
| keras       | bool           | False            | 启用导出到Keras格式，为TensorFlow SavedModel提供兼容性，支持TensorFlow服务和API。                   |
| optimize    | bool           | False            | 在导出到TorchScript时应用优化，适用于移动设备，可能减小模型大小并提高性能。                          |
| half        | bool           | False            | 启用FP16（半精度）量化，减小模型大小，并可能在支持的硬件上加速推理。                                 |
| int8        | bool           | False            | 启用INT8量化，进一步压缩模型并加速推理，几乎不损失精度，主要适用于边缘设备。                        |
| dynamic     | bool           | False            | 允许ONNX和TensorRT导出的动态输入大小，增强处理不同图像尺寸的灵活性。                                 |
| simplify    | bool           | False            | 简化ONNX导出的模型图，可能提高性能和兼容性。                                                         |
| opset       | int            | None             | 指定ONNX opset版本，以兼容不同的ONNX解析器和运行时。如果未设置，使用支持的最新版本。                 |
| workspace   | float          | 4.0              | 为TensorRT优化设置最大工作空间大小（GB），平衡内存使用和性能。                                        |
| nms         | bool           | False            | 为CoreML导出添加非极大值抑制（NMS），对于精确和高效的检测后处理至关重要。                            |
| batch       | int            | 1                | 指定导出模型批量推理大小或导出模型在预测模式下同时处理的最大图像数量。                               |

调整这些参数允许根据特定需求自定义导出过程，如部署环境、硬件限制和性能目标。选择合适的格式和设置对于实现模型大小、速度和准确性之间的最佳平衡至关重要。

# 导出格式
下表列出了可用的YOLOv8导出格式。您可以使用`format`参数导出到任何格式，例如`format='onnx'`或`format='engine'`。导出模型完成后，您可以直接在导出的模型上进行预测或验证，例如`yolo predict model=yolov8n.onnx`。导出完成后，将显示您的模型的使用示例。

| 格式             | `format` 参数   | 模型                          | 元数据 | 参数                               |
|------------------|-----------------|-------------------------------|--------|------------------------------------|
| PyTorch          | -               | yolov8n.pt                    | ✅      | -                                  |
| TorchScript      | `torchscript`   | yolov8n.torchscript           | ✅      | imgsz, optimize, batch             |
| ONNX             | `onnx`          | yolov8n.onnx                  | ✅      | imgsz, half, dynamic, simplify, opset, batch |
| OpenVINO         | `openvino`      | yolov8n_openvino_model/       | ✅      | imgsz, half, int8, batch           |
| TensorRT         | `engine`        | yolov8n.engine                | ✅      | imgsz, half, dynamic, simplify, workspace, batch |
| CoreML           | `coreml`        | yolov8n.mlpackage             | ✅      | imgsz, half, int8, nms, batch      |
| TF SavedModel    | `saved_model`   | yolov8n_saved_model/          | ✅      | imgsz, keras, int8, batch          |
| TF GraphDef      | `pb`            | yolov8n.pb                    | ❌      | imgsz, batch                       |
| TF Lite          | `tflite`        | yolov8n.tflite                | ✅      | imgsz, half, int8, batch           |
| TF Edge TPU      | `edgetpu`       | yolov8n_edgetpu.tflite        | ✅      | imgsz, batch                       |
| TF.js            | `tfjs`          | yolov8n_web_model/            | ✅      | imgsz, half, int8, batch           |
| PaddlePaddle     | `paddle`        | yolov8n_paddle_model/         | ✅      | imgsz, batch                       |
| NCNN             | `ncnn`          | yolov8n_ncnn_model/           | ✅      | imgsz, half, batch                 |

