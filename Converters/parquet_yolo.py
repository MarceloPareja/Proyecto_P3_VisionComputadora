import pandas as pd
import os
from PIL import Image
import io

df = pd.read_parquet('tus_datos.parquet')
output_img_dir = 'dataset/images/train'
output_lbl_dir = 'dataset/labels/train'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

for i, row in df.iterrows():
    # 1. Guardar Imagen
    img = Image.open(io.BytesIO(row['image_bytes']))
    img_width, img_height = img.size
    img_filename = f"img_{i}.jpg"
    img.save(os.path.join(output_img_dir, img_filename))
    
    # 2. Preparar Label
    # Asumiendo que tienes columnas 'x_center', 'y_center', 'width', 'height' y 'keypoints'
    with open(os.path.join(output_lbl_dir, f"img_{i}.txt"), 'w') as f:
        # Normalizar si no lo están
        bn_x = row['x_center'] / img_width
        bn_y = row['y_center'] / img_height
        bn_w = row['width'] / img_width
        bn_h = row['height'] / img_height
        
        line = f"0 {bn_x} {bn_y} {bn_w} {bn_h}" # 0 es el índice de clase
        
        # Añadir Keypoints (también normalizados)
        for kp in row['keypoints']: # Ejemplo: [[x1, y1, v1], [x2, y2, v2]...]
            nx = kp[0] / img_width
            ny = kp[1] / img_height
            v = kp[2]
            line += f" {nx} {ny} {v}"
        
        f.write(line)