import torch, vigia_argus
from ultralytics import YOLO

def test_forward_and_export():
    m = YOLO(vigia_argus.model_yaml("argus_v8x.yaml"))
    _ = m.model(torch.zeros(1,3,640,640))   # forward smoke
    m.export(format="onnx", opset=12, imgsz=640)