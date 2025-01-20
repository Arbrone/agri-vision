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
        self.position = Point(300,300)
        self.direction = np.pi/2
        self.fov = np.zeros((100, 100, 3))
        self.fov_angle = fov_angle
        self.fov_distance = fov_distance
    
    def move(self, direction):
        if direction == "down":
            self.position.y = self.position.y + self.height

    def update_fov(self, new_fov):
        self.fov = new_fov
    
    def get_fov_coord(self, playground_shape):
        # Calculer les distances des points au point de base (x, y)
        # Base du cône, ici en bas / en haut du robot
        xA,yA = self.position.x+self.width/2, self.position.y+self.height
        
        xB = xA + self.fov_distance * np.cos(self.fov_angle)
        yB = yA + self.fov_distance * np.sin(self.fov_angle)

        alpha = np.arctan2(yB-yA, xB-xA)
        beta = alpha + self.fov_angle/2
        xC = xA + self.fov_distance*np.cos(beta)
        yC = yA + self.fov_distance*np.sin(beta)

        beta = alpha - self.fov_angle/2
        xD = xA + self.fov_distance*np.cos(beta)
        yD = yA + self.fov_distance*np.sin(beta)

        return((xA,yA),(xB,yB), (xC,yC), (xD, yD))
    
    def extract_fov(self, playground):
        height, width, _ = playground.shape
        pA, pB, pC, pD = self.get_fov_coord(playground.shape)
        # Créer une grille de coordonnées
        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - pA[0])**2 + (Y - pA[1])**2)

        # Calculer les angles des points par rapport à (x, y)
        angles = np.arctan2(Y - pA[1], X - pA[0])

        # Normaliser les angles par rapport à l'angle de direction du cône
        angle_diff = np.abs(angles - self.direction)
        angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)

        # Créer un masque pour les points dans le cône de vision
        mask = (dist <= self.fov_distance) & (angle_diff <= self.fov_angle / 2)

        # Appliquer le masque pour obtenir les données dans le cône
        cone_data = np.zeros_like(playground)
        cone_data[mask] = playground[mask]

        # # cone_data = cone_data[yA:int(yB), int(xC):int(xD), :]
        cone_data = cone_data[int(pA[1]):int(pB[1]), max(0, int(pC[0])):min(int(pD[0]),640)+1, :]
        cv2.imwrite("fov.jpg", cone_data)

        # fov_image = np.zeros((2*self.fov_distance, 2*self.fov_distance, 3), dtype=np.uint8)
        # fov_image[:cone_data.shape[0], (fov_image.shape[1]//2)-(cone_data.shape[1]//2):(fov_image.shape[1]//2)+(cone_data.shape[1]//2), :] = cone_data
        return cone_data

    def fov_analysis(self):
        results = self.model.predict(self.fov)
        return self.plot_bboxes(results, self.fov)
    
    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy

            for xyxy in xyxys:
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])), (255,0,0), 1)

        return frame