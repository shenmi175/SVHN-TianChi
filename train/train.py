from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':  # 并行处理
    # Load a model
    model = YOLO("v8/yolov8m.yaml")
    model = YOLO("runs/detect/train/weights/best.pt")
    # model = YOLO('yolov8x-oiv7.pt')

    # 初始化TensorBoard Summary Writer
    writer = SummaryWriter('runs/yolov8_training')


    model.train(
                # 训练设置
                data="v8/coco8.yaml",
                epochs=150,  # 训练周期
                imgsz=320,  # 图片大小
                device=0,   # GPU
                verbose=True,   # 详细信息
                resume=True,  # 接着训练
                save=True,  # 保存结果
                patience=100,    # 停止周期数
                plots=True, # 训练集与验证集性能

                # 数据增强设置
                # hsv_h=0.005,
                # hsv_s=0.7,
                # hsv_v=0.5,
                # degrees=25,
                # translate=0.15,
                # scale=0.5,
                # shear=0.05,
                # perspective=0.0,
                # flipud=0.0,
                # fliplr=0.0,
                # bgr=0.05,
                # mosaic=0.9,
                # mixup=0.01,
                # copy_paste=0.01,
                # erasing=0.1,
                # crop_fraction=0.5
                )
    # model.val()

    # 训练结束后关闭SummaryWriter
    writer.close()