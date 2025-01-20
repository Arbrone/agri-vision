import sys
import time
import utils
import numpy as np
import robot
import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLineEdit, QMainWindow,
                               QPushButton, QVBoxLayout, QWidget, QLabel)
from PySide6.QtCharts import QChartView, QPieSeries, QChart


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.robot = robot.Robot("/home/thomas/Workspace/weed_cleaner/model/best.pt")

        # Setup
        self._playground = utils.get_playground("/home/thomas/Workspace/weed_cleaner/model/data/yolo/images/val", (4,4))
        self._playground_robot = self._playground
        self._pixmap = utils.convert_to_pixmap(self._playground)

        # Top Left
        self.map = QLabel()
        self.map.setPixmap(self._pixmap)

        # Top Right
        self.inference = QLabel("Inference time : 33 ms")
        self.confiance = QLabel("Confiance : 86%")

        self.fov = QLabel()
        self.fov.setPixmap(utils.convert_to_pixmap(np.zeros((200,200,3), dtype=np.uint8)))

        # Bottom
        self.start = QPushButton("Start")
        self.reset = QPushButton("Reset")

        self.top_right = QVBoxLayout()
        self.top_right.addWidget(self.inference)
        self.top_right.addWidget(self.confiance)
        self.top_right.addStretch()
        self.top_right.addWidget(self.fov)
        self.top = QHBoxLayout()
        self.top.addWidget(self.map)
        self.top.addLayout(self.top_right)

        self.bottom = QHBoxLayout()
        self.bottom.addWidget(self.start)
        self.bottom.addWidget(self.reset)

        # QWidget Layout
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.top)
        self.layout.addLayout(self.bottom)

        # Signals and Slots
        self.start.clicked.connect(self.start_cleaning)
        self.reset.clicked.connect(self.reset_env)

        self.reset_env()
        self.start_cleaning()


    @Slot()
    def start_cleaning(self):
        pass


    @Slot()
    def reset_env(self):
        self._playground = utils.get_playground("/home/thomas/Workspace/weed_cleaner/model/data/yolo/images/val", (4,4))
        self._pixmap = utils.convert_to_pixmap(self._playground)
        self.map.setPixmap(self._pixmap)
        self.update_map()


    def update_map(self):
        self._playground[self.robot.position.x:self.robot.position.x+self.robot.width, self.robot.position.y:self.robot.position.y+self.robot.height, :] = self.robot.asset
        self._pixmap = utils.convert_to_pixmap(self._playground)
        self.map.setPixmap(self._pixmap)

        #fov_mask = self.get_fov(np.pi/2, np.pi/2)
        fov_mask = self.robot.extract_fov(self._playground)
        self.robot.update_fov(fov_mask)
        result = self.robot.fov_analysis()
        fov_pixmap = utils.convert_to_pixmap(result)
        fov_pixmap = fov_pixmap.scaled(200,200,Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.fov.setPixmap(fov_pixmap)


    def get_fov(self, angle, direction_angle):
        max_distance = 150
        height, width, _ = self._playground.shape

        # Créer une grille de coordonnées
        Y, X = np.ogrid[:height, :width]

        # Calculer les distances des points au point de base (x, y)
        x,y = self.robot.position[0]+self.robot.height/2, self.robot.position[1]+self.robot.width
        dist = np.sqrt((X - x)**2 + (Y - y)**2)

        # Calculer les angles des points par rapport à (x, y)
        angles = np.arctan2(Y - y, X - x)

        # Normaliser les angles par rapport à l'angle de direction du cône
        angle_diff = np.abs(angles - direction_angle)
        angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)

        # Créer un masque pour les points dans le cône de vision
        mask = (dist <= max_distance) & (angle_diff <= angle / 2)

        # Appliquer le masque pour obtenir les données dans le cône
        cone_data = np.zeros_like(self._playground)
        cone_data[mask] = self._playground[mask]
        cv2.imwrite("mask.jpg", cone_data)
        
        xA = x
        yA = y
        xB = x + max_distance * np.cos(angle)
        yB = y + max_distance * np.sin(angle)

        alpha = np.arctan2(yB-yA, xB-xA)
        beta = alpha + angle/2
        xC = xA + max_distance*np.cos(beta)
        yC = yA + max_distance*np.sin(beta)

        beta = alpha - angle/2
        xD = xA + max_distance*np.cos(beta)
        yD = yA + max_distance*np.sin(beta)

        # cone_data = cone_data[yA:int(yB), int(xC):int(xD), :]
        cone_data = cone_data[yA:int(yB), max(0, int(xC)):min(int(xD),640)+1, :]
        cv2.imwrite("fov.jpg", cone_data)
        print(((xA,yA),(xB,yB), (xC,yC), (xD, yD)))
        print(cone_data.shape)
        fov_image = np.zeros((2*max_distance, 2*max_distance, 3), dtype=np.uint8)
        fov_image[:cone_data.shape[0], (fov_image.shape[1]//2)-(cone_data.shape[1]//2):(fov_image.shape[1]//2)+(cone_data.shape[1]//2), :] = cone_data
        return fov_image


class MainWindow(QMainWindow):
    def __init__(self, widget):
        super().__init__()
        self.setWindowTitle("AgriVision")
        self.setCentralWidget(widget)


if __name__ == "__main__":
    # Qt Application
    app = QApplication(sys.argv)
    widget = Widget()
    window = MainWindow(widget)
    #window.setFixedSize(860, 700)
    window.setFixedSize(1000, 700)
    window.show()

    # Execute application
    sys.exit(app.exec())