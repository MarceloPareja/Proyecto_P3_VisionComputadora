from ultralytics import YOLO
import cv2

#Carga de los modelos ya entrenados
def main():
    layer2_model = YOLO("layer2-model.pt")
    layer3_model = YOLO("layer3-model.pt")

    cap = cv2.VideoCapture("Videos/dogs1.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Realiza la detección de poses con el modelo de la capa 2
        results1 = layer2_model(frame, conf=0.4)[0]
        boxes = results1.boxes
        keypoints = results1.keypoints
        DOG_CLASS_ID = 0  # según tu YAML

        for i, cls in enumerate(boxes.cls):
            if int(cls) != DOG_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue
            results2 = layer3_model(crop, conf=0.3)[0]
            if results2.keypoints is None or len(results2.keypoints.xy) == 0:
                continue  # no hubo detección en capa 2

            kp = kp = results2.keypoints.xy[0].clone()

            for x, y in kp:
                x += x1
                y += y1
            for x, y in kp:
                cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("Pipeline Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()