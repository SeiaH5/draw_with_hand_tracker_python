import cv2
import numpy as np

class GestureApp:
    COLOURS = [
        (0,   0,   0),    # 0 black
        (255, 0,   0),    # 1 blue
        (0,   255, 0),    # 2 green
        (0,   0,   255),  # 3 red
        (0,   255, 255),  # 4 yellow
        (0,   165, 255),  # 5 orange
        (128, 0,   128),  # 6 purple
        (203, 192, 255),  # 7 pink
        (255, 255, 0),    # 8 cyan
        (255, 255, 255),  # 9 white
    ]

    def __init__(self, width=1280, height=720):
        self.vid_width   = width
        self.vid_height  = height

        self.video = cv2.VideoCapture(0)
        self.video.set(3, self.vid_width)
        self.video.set(4, self.vid_height)

        self.imageCanvas     = np.zeros((self.vid_height, self.vid_width, 3), np.uint8)
        self.curr_colour     = 0
        self.pen_thickness   = 8
        self.erase_thickness = 30
        self.canWrite        = False
        self.eraseMode       = False
        self.running         = True

    def reset(self):
        self.imageCanvas     = np.zeros((self.vid_height, self.vid_width, 3), np.uint8)
        self.curr_colour     = 0
        self.pen_thickness   = 8
        self.erase_thickness = 30
        self.canWrite        = False
        self.eraseMode       = False
        self.running         = True

    def clearCanvas(self):
        self.imageCanvas = np.zeros((self.vid_height, self.vid_width, 3), np.uint8)

    def compositeCanvas(self, background):
        """Blend the drawing canvas on top of a background image."""
        imageCanvasNot = cv2.bitwise_not(self.imageCanvas)
        imageCanvasNot[np.all(imageCanvasNot == (0, 0, 0), axis=-1)] = self.COLOURS[self.curr_colour]
        return np.where(self.imageCanvas == (0, 0, 0), background, imageCanvasNot)

    def drawHUD(self, image):
        mode_text = "DRAW" if self.canWrite else ("ERASE" if self.eraseMode else "IDLE")
        cv2.putText(image, f"Mode: {mode_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        thickness_val = self.pen_thickness if self.canWrite else self.erase_thickness
        cv2.putText(image, f"Size: {thickness_val}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(image, (10, 75), (40, 105), self.COLOURS[self.curr_colour], cv2.FILLED)

    def release(self):
        self.video.release()
        cv2.destroyAllWindows()