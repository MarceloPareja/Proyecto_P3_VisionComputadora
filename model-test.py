from ultralytics import YOLO
import time

# Load a model
#model = YOLO("yolo26n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo26n-pose.yaml").load("yolo26n-pose.pt")  # build from YAML and transfer weights

# Test the model
for r in model.track(source="0", stream=True, show=True, persist=True): pass  # webcam
