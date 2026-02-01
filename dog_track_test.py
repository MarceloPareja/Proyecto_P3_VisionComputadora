from ultralytics import YOLO

def main():
    #Importa el modelo de poses
    model = YOLO("runs/pose/train3/weights/best.pt")


    # Probar el modelo con un video
    results = model.track(source="Images/", save=True, persist=True)
    # Valida el modelo e imprime métricas
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()