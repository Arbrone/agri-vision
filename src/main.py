import sys
import time
import utils
import numpy as np
import robot
import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLineEdit, QMainWindow,
                               QPushButton, QVBoxLayout, QWidget, QLabel)
from PySide6.QtCharts import QChartView, QPieSeries, QChart


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.robot = robot.Robot("/home/thomas/Workspace/weed_cleaner/model/best_small.pt")

        # Setup
        self.playground = utils.get_playground("/home/thomas/Workspace/weed_cleaner/model/data/yolo/images/val", (4,4))
        self.playground_robot = self.playground
        self.pixmap = utils.convert_to_pixmap(self.playground)

        # Top Left
        self.map = QLabel()
        self.map.setPixmap(self.pixmap)

        # Top Right
        # self.inference = QLabel(f"Inference time : {33}ms")
        # self.confiance = QLabel(f"Confiance : {86}%")
        self.inference = QLabel("")
        self.confiance = QLabel("")
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

        # Shortcuts
        ## GUI
        self.quit_shortcut = QShortcut(Qt.Key.Key_Q, self)
        self.quit_shortcut.activated.connect(QApplication.quit)
        self.reset_shortcut = QShortcut(Qt.Key.Key_R, self)
        self.reset_shortcut.activated.connect(self.reset_env)

        ## Robot
        self.move_up_shortcut = QShortcut(Qt.Key.Key_Up, self)
        self.move_up_shortcut.activated.connect(lambda: self.move("up"))
        self.move_down_shortcut = QShortcut(Qt.Key.Key_Down, self)
        self.move_down_shortcut.activated.connect(lambda: self.move("down"))
        self.move_left_shortcut = QShortcut(Qt.Key.Key_Left, self)
        self.move_left_shortcut.activated.connect(lambda: self.move("left"))
        self.move_right_shortcut = QShortcut(Qt.Key.Key_Right, self)
        self.move_right_shortcut.activated.connect(lambda: self.move("right"))

        self.reset_env()
        self.start_cleaning()


    @Slot()
    def start_cleaning(self):
        pass

    @Slot()
    def move(self, direction):
        print(direction)
        self.robot.move(direction)
        self.update_map()

    @Slot()
    def reset_env(self):
        self.playground = utils.get_playground("/home/thomas/Workspace/weed_cleaner/model/data/yolo/images/val", (4,4))
        self.playground_robot = self.playground
        self.pixmap = utils.convert_to_pixmap(self.playground)
        self.map.setPixmap(self.pixmap)

        self.update_map()


    def update_map(self):
        playground_copy = self.playground_robot.copy()

        playground_copy[self.robot.position.y:self.robot.position.y+self.robot.height, 
                        self.robot.position.x:self.robot.position.x+self.robot.width, :] = self.robot.asset

        playground_copy = cv2.circle(playground_copy, 
                                    (int(self.robot.fov_base.x), int(self.robot.fov_base.y)), 
                                    3, (255, 0, 0), 2)
        
        p1, p2 = self.robot.get_fov_coord()
        
        playground_copy = cv2.line(playground_copy, 
                                (int(self.robot.fov_base.x), int(self.robot.fov_base.y)), 
                                (int(p1.x), int(p1.y)), (0, 0, 255), 2)
        playground_copy = cv2.line(playground_copy, 
                                (int(self.robot.fov_base.x), int(self.robot.fov_base.y)), 
                                (int(p2.x), int(p2.y)), (0, 0, 255), 2)

        self.pixmap = utils.convert_to_pixmap(playground_copy)        
        self.map.setPixmap(self.pixmap)

        fov_mask = self.robot.extract_fov(self.playground)
        self.robot.update_fov(fov_mask)

        result = self.robot.fov_analysis()

        fov_pixmap = utils.convert_to_pixmap(result)
        fov_pixmap = fov_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.fov.setPixmap(fov_pixmap)


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
    window.setFixedSize(860, 700)
    #window.setFixedSize(1000, 700)
    window.show()

    # Execute application
    sys.exit(app.exec())