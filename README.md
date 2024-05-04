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

**注意：**部分库是没有必要安装的，你可以在使用命令前手动打开文件进行删除

### 进行训练

进入train.py文件，修改文件路径与coco.yaml文件中的数据集路径。如有必要，请参考yolov8官方文档。

使用已经训练好的预训练权重best.pt，该模型在原始训练集（30000张图片）上训练100个周期，其结果如下：
![results](train\runs\detect\train\results.png)



## 最终排名

![2024-4-29-分数-排名](data\2024-4-29-分数-排名.jpg)
