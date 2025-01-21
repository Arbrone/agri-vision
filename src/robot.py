import cv2
import numpy as np

from utils import Point
from ultralytics import YOLO

class Robot:
    def __init__(self, model, fov_angle=np.pi/2, fov_distance=100):
        self.asset = cv2.imread("/home/thomas/Workspace/weed_cleaner/assets/robot.jpg")
        self.height = self.asset.shape[0]
        self.width = self.asset.shape[1]
        self.model = YOLO(model)
        self.position = Point(0,0)
        self.direction = np.pi/2
        self.fov = np.zeros((100, 100, 3))
        self.fov_angle = fov_angle
        self.fov_base = Point(self.position.x+self.width/2, self.position.y+self.height)
        self.fov_distance = fov_distance

    def reset(self):
        self.position = Point(0,0)
        self.direction = np.pi/2
        self.fov_angle = np.pi/2
        self.fov_base = Point(self.position.x+self.width/2, self.position.y+self.height)

    def check_position(self, position):
        return position

    def move(self, direction):
        match direction:
            case "down":
                print(self.position.x, self.position.y)
                self.position.y = self.check_position(self.position.y + self.height)
                self.direction = np.pi/2
                self.fov_base = Point(self.position.x+self.width/2, self.position.y+self.height)
            case "up":
                self.position.y = self.check_position(self.position.y - self.height)
                self.direction = -np.pi/2
                self.fov_base = Point(self.position.x+self.width/2, self.position.y)
            case "left":
                self.position.x = self.check_position(self.position.x - self.width)
                self.direction = np.pi
                self.fov_base = Point(self.position.x, self.position.y+self.height/2)
            case "right":
                self.position.x = self.check_position(self.position.x + self.width)
                self.direction = 0
                self.fov_base = Point(self.position.x+self.width, self.position.y+self.height/2)


    def update_fov(self, new_fov):
        self.fov = new_fov
    
    def get_fov_coord(self):
        delta_angle = np.pi / 4  # 45 deg

        angle1 = self.direction + delta_angle
        angle2 = self.direction - delta_angle

        x1 = self.fov_base.x + self.fov_distance * np.cos(angle1)
        y1 = self.fov_base.y + self.fov_distance * np.sin(angle1)

        x2 = self.fov_base.x + self.fov_distance * np.cos(angle2)
        y2 = self.fov_base.y + self.fov_distance * np.sin(angle2)

        return Point(x1, y1), Point(x2, y2)
    
    def extract_fov(self, playground):
        height, width, _ = playground.shape
        p1, p2 = self.get_fov_coord()

        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - self.fov_base.x)**2 + (Y - self.fov_base.y)**2)

        angles = np.arctan2(Y - self.fov_base.y, X - self.fov_base.x)

        angle_diff = np.abs(angles - self.direction)
        angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)

        mask = (dist <= self.fov_distance) & (angle_diff <= self.fov_angle / 2)

        cone_data = np.zeros_like(playground)
        cone_data[mask] = playground[mask]

        cone_boundaries = np.array([
                                    [int(self.fov_base.x), int(self.fov_base.y)],
                                    [int(p1.x), int(p1.y)],
                                    [int(p2.x), int(p2.y)],
                                    [int(self.fov_base.x + self.fov_distance * np.cos(self.direction)), int(self.fov_base.y + self.fov_distance * np.sin(self.direction))]
                                    ])

        x,y,w,h = cv2.boundingRect(cone_boundaries)
        return cone_data[max(0,y):min(height,y+h), max(0,x):min(height,x+w), :]

    def fov_analysis(self):
        results = self.model.predict(self.fov, conf=0.4)
        return self.plot_bboxes(results)
    
    def plot_bboxes(self, results):
        img = None
        for result in results:
            img = result.plot(line_width=1, labels=False, probs=True)
            print(result.boxes.conf)
        return img