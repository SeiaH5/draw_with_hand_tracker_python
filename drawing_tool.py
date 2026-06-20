import cv2
import math
from gui import GestureApp

class DrawingTool(GestureApp):
    """
    Handles drawing and erasing strokes on the canvas.
    Inherits camera/canvas state from GestureApp.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index_lx = None
        self.index_ly = None

    def reset(self):
        super().reset()
        self.index_lx = None
        self.index_ly = None

    @staticmethod
    def pointDistance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def drawStroke(self, position, colour, thickness):
        cv2.circle(self.imageCanvas, position, thickness // 2, colour, cv2.FILLED)
        if self.index_lx is not None and self.index_ly is not None:
            cv2.line(self.imageCanvas, (self.index_lx, self.index_ly), position, colour, thickness)

    def applyDraw(self, index_cx, index_cy):
        self.drawStroke((index_cx, index_cy), (255, 255, 255), self.pen_thickness)
        self.index_lx, self.index_ly = index_cx, index_cy

    def applyErase(self, index_cx, index_cy):
        self.drawStroke((index_cx, index_cy), (0, 0, 0), self.erase_thickness)
        self.index_lx, self.index_ly = index_cx, index_cy

    def resetLastPos(self):
        self.index_lx = None
        self.index_ly = None

    def adjustSize(self, increase: bool):
        if self.canWrite:
            if increase:
                self.pen_thickness += 6
            elif self.pen_thickness - 6 >= 4:
                self.pen_thickness -= 6
        elif self.eraseMode:
            if increase:
                self.erase_thickness += 6
            elif self.erase_thickness - 6 >= 4:
                self.erase_thickness -= 6