from ultralytics import YOLO

def main():
    #Importa el modelo de poses
    model = YOLO("yolo11n-pose.pt")

    #Entrenar el modelo con los conjuntos de datos
    results = model.train(data="Datasets/Capa2DetecctionPose/dataset-layer2.yaml",
                           epochs=50,
                             imgsz=640,
                               device="0",
                               name="layer2-model")

    trained = YOLO("runs/pose/layer2-model/weights/best.pt")

    # Probar el modelo con un video
    trained.track(source="/Videos/dogs1.mp4", show=True, save=True)

    # Validar el modelo e imprimir métricas
    metrics = trained.val(data="Datasets/Capa2DetecctionPose/dataset-layer2.yaml")
    print(metrics)


if __name__ == "__main__":
    main()