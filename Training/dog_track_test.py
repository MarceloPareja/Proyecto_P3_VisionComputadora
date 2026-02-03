from ultralytics import YOLO

def main():
    #Importa el modelo de poses
    model = YOLO("layer3-model.pt")


    # Probar el modelo con un video
    results = model.track(source="Videos/dogbite.mp4", show=True, save=True)
    # Valida el modelo e imprime métricas
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()