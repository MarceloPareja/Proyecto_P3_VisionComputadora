from ultralytics import YOLO

def main():
    #Importa el modelo de clasificación
    model = YOLO("yolo11n-cls.pt") 
    #Entrenar el modelo con los conjuntos de datos
    model.train(data="Datasets/Capa4ClasificaEmocion/images",
                           epochs=75,
                             imgsz=256,
                             batch=32,
                               device="0",
                               name="emotion-detect-model")

    trained = YOLO("runs/classify/emotion-detect-model/weights/best.pt")

    # Probar el modelo con un video
    trained(source="/Videos/dogs1.mp4", show=True, save=True)

    # Validar el modelo e imprimir métricas
    metrics = trained.val(data="Datasets/Capa4ClasificaEmocion/images")
    print(metrics)


if __name__ == "__main__":
    main()