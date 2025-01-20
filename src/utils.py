import numpy as np
import cv2
import random
import glob
from pathlib import Path
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def get_playground(root_dir, shape):
    images_path = glob.glob(str(Path(root_dir,'*.jpg')))
    images = random.choices(images_path, k=shape[0]*shape[1])
    
    playground = np.zeros((640 * shape[0], 640 * shape[1], 3), dtype=np.uint8)

    for idx, path in enumerate(images):
        image = cv2.imread(path)
        if image is None:
            continue

        row = idx // shape[1]
        col = idx % shape[1]

        x_start = row * 640
        y_start = col * 640
        playground[x_start:x_start+640, y_start:y_start+640, :] = image
    
    playground = cv2.resize(playground, (640, 640), interpolation = cv2.INTER_LINEAR)
    cv2.imwrite("playground.jpg", playground)
    return playground
    

def convert_to_pixmap(image):
    height, width, channel = image.shape
    bytes_per_line = channel * width

    image_array_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimage = QImage(image_array_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

    return QPixmap.fromImage(qimage)