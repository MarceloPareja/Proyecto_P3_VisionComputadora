from ultralytics import YOLO

#Importa el modelo de poses
model = YOLO("yolo26n-pose.pt")

#Entrenar el modelo con los conjuntos de datos
results = model.train(data="dog-pose.yaml", epochs=1, imgsz=640)

# Probar el modelo con un video
results = model.track(source="dogs1.mp4", show=True, save=True)

# Valida el modelo e imprime métricas
metrics = model.val()
print(metrics)