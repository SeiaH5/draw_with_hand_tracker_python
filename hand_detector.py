import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from file_manager import FileManager

MODEL_PATH = "hand_landmarker.task"

class HandDetector(FileManager):
    FINGER_TIP_IDS = [4, 8, 12, 16, 20]

    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector     = mp_vision.HandLandmarker.create_from_options(options)
        self.landmark_pts = []
        self.fingersUp    = [0] * 5

    def detect(self, imageWebcam):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(imageWebcam, cv2.COLOR_BGR2RGB)
        )
        return self.detector.detect(mp_image)

    def updateLandmarks(self, hand_landmarks, image_w, image_h):
        self.landmark_pts = []
        self.fingersUp    = [0] * 5
        for lm in hand_landmarks:
            self.landmark_pts.append([int(lm.x * image_w), int(lm.y * image_h)])
        self._checkFingersUp()

    def _checkFingersUp(self):
        palm = self.landmark_pts[0]
        for i in range(5):
            tip = self.landmark_pts[self.FINGER_TIP_IDS[i]]
            pip = self.landmark_pts[self.FINGER_TIP_IDS[i] - 1]
            dip = self.landmark_pts[self.FINGER_TIP_IDS[i] - 2]
            if i == 0:
                if self.pointDistance(tip, palm) > self.pointDistance(pip, palm):
                    self.fingersUp[0] = 1
            else:
                if self.pointDistance(tip, palm) > self.pointDistance(dip, palm):
                    self.fingersUp[i] = 1

    def checkFingersDown(self, indices):
        return all(self.fingersUp[i] == 0 for i in indices)

    def checkFingerVerticalUp(self, finger, threshold):
        tip = self.landmark_pts[self.FINGER_TIP_IDS[finger]]
        mcp = self.landmark_pts[self.FINGER_TIP_IDS[finger] - 2]
        return math.fabs(tip[0] - mcp[0]) <= threshold and tip[1] < mcp[1]

    def checkFingersHorizontal(self):
        for i in range(1, 5):
            tip = self.landmark_pts[self.FINGER_TIP_IDS[i]]
            pip = self.landmark_pts[self.FINGER_TIP_IDS[i] - 2]
            if math.fabs(tip[1] - pip[1]) >= 20:
                return False
        return True

    def posOnScreen(self, p):
        return 0 <= p[0] <= self.vid_width and 0 <= p[1] <= self.vid_height

    def checkThumbsUp(self):
        return (
            all(self.posOnScreen(self.landmark_pts[f]) and
                self.posOnScreen(self.landmark_pts[f - 2])
                for f in self.FINGER_TIP_IDS)
            and self.checkFingerVerticalUp(0, 20)
            and self.fingersUp[0]
            and self.checkFingersDown([1, 2, 3, 4])
            and self.checkFingersHorizontal()
        )

    def drawLandmarks(self, image):
        pts = self.landmark_pts
        for a, b in self.HAND_CONNECTIONS:
            cv2.line(image, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
        for i, pt in enumerate(pts):
            colour = (0, 0, 255) if i in self.FINGER_TIP_IDS else (255, 0, 0)
            cv2.circle(image, tuple(pt), 5, colour, cv2.FILLED)