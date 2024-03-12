import torch
from pathlib import Path

from ultralytics import YOLO


# Your other imports and code for setting up the training, etc.
import torch
import torch.onnx
import torchvision


def main():
    # 调用训练函数，例如：
    # train_model()
    model = YOLO('G:/YOLOv8-multi-task/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml',
                 task='multi')  # build a new model from YAML




    model.train(data='G:/YOLOv8-multi-task/ultralytics/datasets/bdd-multi1.yaml',workers=4, batch=20, epochs=3,
                imgsz=(640, 640), device=0, name='yolopm', val=True, task='multi', classes=[2, 3, 4, 9, 10, 11],
                combine_class=[2, 3, 4, 9], single_cls=True)

    pass

if __name__ == '__main__':
    # Windows平台下确保多进程的安全性
    torch.multiprocessing.freeze_support()
    main()  # 调用主训练函数
    # tensorboard - -logdir = "G:\YOLOv8-multi-task\ultralytics\runs\multi\yolopm6"
