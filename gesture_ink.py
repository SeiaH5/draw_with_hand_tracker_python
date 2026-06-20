import cv2
from hand_detector import HandDetector

class GestureInk(HandDetector):
    WIN_NAME = "GestureInk"

    def __init__(self):
        super().__init__()
        self._printControls()

    @staticmethod
    def _printControls():
        print("Controls:")
        print("  W       - toggle draw mode")
        print("  E       - toggle erase mode")
        print("  C       - clear canvas")
        print("  O       - open PDF or image file")
        print("  R       - reset everything")
        print("  , / .   - decrease / increase brush size")
        print("  0-9     - change colour")
        print("  S       - save and quit")
        print("  Q       - quit without saving")
        print("  Thumbs-up gesture - save and quit")
        print("  Pinky-up gesture  - clear canvas")

    def run(self):
        while self.running:
            success, imageWebcam = self.video.read()
            if not success:
                break

            imageWebcam = cv2.flip(imageWebcam, 1)
            image       = self.getBackground(imageWebcam)
            image_h, image_w = image.shape[:2]

            image          = self.compositeCanvas(image)
            imageCanvasNot = cv2.bitwise_not(self.imageCanvas)

            result = self.detect(imageWebcam)

            if result.hand_landmarks:
                self.updateLandmarks(result.hand_landmarks[0], image_w, image_h)

                index_cx = self.landmark_pts[self.FINGER_TIP_IDS[1]][0]
                index_cy = self.landmark_pts[self.FINGER_TIP_IDS[1]][1]

                draw_fingers_down = [2, 3, 4]

                if self.canWrite and self.fingersUp[1] and self.checkFingersDown(draw_fingers_down):
                    self.applyDraw(index_cx, index_cy)
                elif self.eraseMode and self.fingersUp[1] and self.checkFingersDown(draw_fingers_down):
                    self.applyErase(index_cx, index_cy)
                else:
                    self.resetLastPos()

                # Thumbs-up → save & quit
                if not self.canWrite and self.checkThumbsUp():
                    self.saveResults(imageWebcam, imageCanvasNot, image)
                    self.running = False

                # Pinky-up → clear canvas
                if not self.canWrite and self.checkFingersDown([1, 2, 3]) and self.checkFingerVerticalUp(4, 50):
                    self.clearCanvas()

                # Fingertip cursor
                cv2.circle(image,
                           (self.landmark_pts[self.FINGER_TIP_IDS[1]][0],
                            self.landmark_pts[self.FINGER_TIP_IDS[1]][1]),
                           self.erase_thickness // 2, (0, 255, 255), cv2.FILLED)

                self.drawLandmarks(image)
            else:
                self.resetLastPos()

            self.drawHUD(image)
            cv2.imshow(self.WIN_NAME, image)
            cv2.moveWindow(self.WIN_NAME, 0, 0)

            self._handleKeys(imageWebcam, imageCanvasNot, image)

        self.release()

    def _handleKeys(self, imageWebcam, imageCanvasNot, image):
        keys = cv2.waitKey(1) & 0xFF

        if keys == ord('q'):
            self.running = False
        elif keys == ord('s'):
            self.saveResults(imageWebcam, imageCanvasNot, image)
            self.running = False
        elif keys == ord('w'):
            self.canWrite = not self.canWrite
            if self.canWrite:
                self.eraseMode = False
        elif keys == ord('e'):
            self.eraseMode = not self.eraseMode
            if self.eraseMode:
                self.canWrite = False
        elif keys == ord('c'):
            self.clearCanvas()
        elif keys == ord('r'):
            self.reset()
        elif keys == ord(','):
            self.adjustSize(increase=False)
        elif keys == ord('.'):
            self.adjustSize(increase=True)
        elif ord('0') <= keys <= ord('9'):
            self.curr_colour = int(chr(keys))
        elif keys == ord('o'):
            self.openFile()