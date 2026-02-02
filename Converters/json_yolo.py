import json
import os
import cv2

# Cargar JSON
with open(r"Datasets/Capa2DetecctionPose/keypoints.json") as f:
    coco = json.load(f)

# Images: dict {id: {file_name: ...}}
images = {
    int(img_id): img_data
    for img_id, img_data in coco["images"].items()
}

# Crear carpeta de labels
labels_dir = r"Datasets/Capa2DetecctionPose/labels"
os.makedirs(labels_dir, exist_ok=True)

images_dir = r"Datasets/Capa2DetecctionPose/images/images"

# Procesar anotaciones
for ann in coco["annotations"]:
    image_id = int(ann["image_id"])
    file_name = images[image_id]
    image_path = os.path.join(images_dir, file_name)

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ No se pudo leer {image_path}")
        continue

    img_h, img_w = img.shape[:2]

    # BBOX (COCO -> YOLO)
    x_min, y_min, x_max, y_max = ann["bbox"]

    bw = x_max - x_min
    bh = y_max - y_min

    x_c = (x_min + bw / 2) / img_w
    y_c = (y_min + bh / 2) / img_h
    bw /= img_w
    bh /= img_h
    # KEYPOINTS
    keypoints = ann["keypoints"]
    kp_norm = []
    for i in range(0, len(keypoints), 1):
        kp_norm.extend([
            keypoints[i][0] / img_w,
            keypoints[i][1] / img_h,
            keypoints[i][2]
        ])

    # Clase (ajusta si no empiezan en 1)
    class_id = ann["category_id"] - 1

    label_line = " ".join(map(str, [class_id, x_c, y_c, bw, bh] + kp_norm))

    label_path = os.path.join(
        labels_dir,
        os.path.splitext(file_name)[0] + ".txt"
    )

    with open(label_path, "a") as f:
        f.write(label_line + "\n")