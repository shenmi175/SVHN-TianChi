本项目出自阿里天池竞赛[零基础入门CV - 街景字符编码识别](https://tianchi.aliyun.com/competition/entrance/531795/information)学习赛。

本项目参考了其他选手的一些处理技巧，使用yolov8框架进行训练。截至2024年4月29日，分数为0.9477，排名5/7496

参考资料：
[动手学CV-Pytorch](https://datawhalechina.github.io/dive-into-cv-pytorch/#/?id=dive-into-cv-pytorch)
[YOLOV8官方文档](https://docs.ultralytics.com/)
[Albumentations库说明文档](https://albumentations.ai/docs/)



## 快速开始

### 快速创建环境

进入environment.yaml文件所在的文件夹，使用以下命令创建环境并安装所需库：

```python
conda env create -f environment.yml
```

注意：部分库是没有必要安装的，你可以在使用命令前手动打开文件进行删除

### 进行训练

进入train.py文件，修改文件路径与coco.yaml文件中的数据集路径。如有必要，请参考yolov8官方文档。

使用已经训练好的预训练权重best.pt，该模型在原始训练集（30000张图片）上训练100个周期，其结果如下：
![results](https://github.com/shenmi175/SVHN-TianChi/assets/102409368/48b5daf7-0800-497d-8ba8-32f226da2e3a)

注意：如果进行了数据扩增（图片数量大于30000张），最好从新使用yolov8m.pt进行训练。


### 输出处理
1. iou=0.5，置信度=0.001
2. 使用官方框架添加全局nms，例如：
```python
from ultralytics import YOLO
model = YOLO('path/to/best.pt')  # 加载自定义训练模型
# 导出模型
model.export(format='coreml',nms=True)
```
4. 识别按照边框位置进行排序
输出处理十分重要，这在很大程度上决定了最终分数。更多方法请参考竞赛论坛。

### 数据增强
1. 验证集部分图片标注存在问题，有部分图片的数字缺少标注，导致模型在验证集上表现不高。建议减少验证集图片数量，添加到训练集中。
2. 部分类别误判率高，尝试额外扩充
3. yolov8自带数据增强处理，分数略微提升
4. 使用Albumentations库进行额外数据扩增



## 最终排名
![2024-4-29-分数-排名](https://github.com/shenmi175/SVHN-TianChi/assets/102409368/52eb72a2-a9a6-4e27-a9bd-c9fac2789cde)


