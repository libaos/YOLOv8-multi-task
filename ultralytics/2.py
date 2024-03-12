#
# import torch.onnx
# # 假定'model.pt'是你的模型权重文件
# checkpoint = torch.load('G:/YOLOv8-multi-task/ultralytics/best.pt')
#
# # 假定checkpoint字典有一个键'model_state_dict'存储模型状态
# model.load_state_dict(checkpoint['model_state_dict'])  # 载入模型状态
#
# model.eval()  # 设置模型到评估模式
# # 加载预训练模型
# # 创建一个样例输入张量。大小和模型输入层的大小一致。
# # 例如，如果你的模型输入是一个1x3x224x224的彩色图像，这里创建相应的张量。
# dummy_input = torch.randn(1, 3, 224, 224)
# # 设置ONNX文件的输出路径
# output_path = 'G:/YOLOv8-multi-task/ultralytics/best.onnx'
# # 导出模型
# # verbose=True将打印模型转换过程中的详细信息
# torch.onnx.export(model, dummy_input, output_path, verbose=True)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model = YOLO('best.pt')

# 导出模型
model.export(format='onnx')
