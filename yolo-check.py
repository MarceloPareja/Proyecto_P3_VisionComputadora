from ultralytics import YOLO
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

model = YOLO("yolo26n.pt")
model.info()
