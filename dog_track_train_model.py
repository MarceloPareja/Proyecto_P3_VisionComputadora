from ultralytics import YOLO

def main():
    #Importa el modelo de poses
    model = YOLO("runs/pose/train3/weights/last.pt")

    #Entrenar el modelo con los conjuntos de datos
    results = model.train(data="dog-pose.yaml", epochs=100, resume=True, imgsz=640, device="0")

    # Probar el modelo con un video
    model.track(source="/Videos/dogs1.mp4", show=True, save=True)

    # Validar el modelo e imprimir métricas
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()