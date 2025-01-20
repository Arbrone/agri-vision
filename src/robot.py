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

        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - self.fov_base.x)**2 + (Y - self.fov_base.y)**2)

        # Calculer les angles des points par rapport à (x, y)
        angles = np.arctan2(Y - self.fov_base.y, X - self.fov_base.x)

        # Normaliser les angles par rapport à l'angle de direction du cône
        angle_diff = np.abs(angles - self.direction)
        angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)

        # Créer un masque pour les points dans le cône de vision
        mask = (dist <= self.fov_distance) & (angle_diff <= self.fov_angle / 2)

        # Appliquer le masque pour obtenir les données dans le cône
        cone_data = np.zeros_like(playground)
        cone_data[mask] = playground[mask]

        # # cone_data = cone_data[yA:int(yB), int(xC):int(xD), :]
        # match self.direction:
        #     case 
        # cone_data = cone_data[int(self.fov_base.y):int(pB[1]), max(0, int(pC[0])):min(int(pD[0]),640)+1, :]
        cv2.imwrite("fov.jpg", cone_data)

        # fov_image = np.zeros((2*self.fov_distance, 2*self.fov_distance, 3), dtype=np.uint8)
        # fov_image[:cone_data.shape[0], (fov_image.shape[1]//2)-(cone_data.shape[1]//2):(fov_image.shape[1]//2)+(cone_data.shape[1]//2), :] = cone_data
        return cone_data

    def fov_analysis(self):
        results = self.model.predict(self.fov)
        return self.plot_bboxes(results)
    
    # def plot_bboxes(self, results, frame):
    #     xyxys = []
    #     confidences = []

    #     for result in results:
    #         boxes = result.boxes.cpu().numpy()
    #         xyxys = boxes.xyxy

    #         for xyxy in xyxys:
    #             cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])), (255,0,0), 1)

    #         return result.plot()

    #     # return frame

    def plot_bboxes(self, results):
        img = None

        for result in results:
            img = result.plot(line_width=1, labels=False, probs=True)

        return img