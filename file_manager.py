import cv2
import numpy as np
import os
import pathlib
from tkinter import filedialog
from pdf2image import convert_from_path
from PIL import Image
from drawing_tool import DrawingTool

class FileManager(DrawingTool):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pdfMode  = False
        self.imgMode  = False
        self.imagePDF = None
        self.imgNPRGB = None

    def reset(self):
        super().reset()
        self.pdfMode = False
        self.imgMode = False

    def openFile(self):
        file = filedialog.askopenfile(
            mode='r',
            filetypes=[
                ("PDF and images", "*.pdf *.png *.jpg *.jpeg *.gif *.bmp *.ico"),
                ("PDF file", "*.pdf"),
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"),
            ]
        )
        if not file:
            return
        filepath = os.path.abspath(file.name)
        suffix   = pathlib.Path(file.name).suffix.lower()
        if suffix == ".pdf":
            pages         = convert_from_path(filepath)
            self.imagePDF = np.array(pages[0])
            self.pdfMode  = True
            self.imgMode  = False
        else:
            imgPic        = Image.open(filepath)
            imgNP         = np.asarray(imgPic)
            self.imgNPRGB = cv2.cvtColor(imgNP, cv2.COLOR_RGB2BGR)
            self.imgMode  = True
            self.pdfMode  = False

    def getBackground(self, imageWebcam):
        if self.pdfMode and self.imagePDF is not None:
            return self._resizeAndPad(self.imagePDF)
        if self.imgMode and self.imgNPRGB is not None:
            return self._resizeAndPad(self.imgNPRGB)
        return imageWebcam.copy()

    def _resizeAndPad(self, img, padColor=0):
        h, w   = img.shape[:2]
        sh, sw = self.vid_height, self.vid_width
        interp = cv2.INTER_AREA if (h > sh or w > sw) else cv2.INTER_CUBIC
        aspect = w / h

        if aspect > 1:
            new_w = sw
            new_h = int(np.round(new_w / aspect))
            pad_vert = (sh - new_h) / 2
            pad_top, pad_bot = int(np.floor(pad_vert)), int(np.ceil(pad_vert))
            pad_left, pad_right = 0, 0
        elif aspect < 1:
            new_h = sh
            new_w = int(np.round(new_h * aspect))
            pad_horz = (sw - new_w) / 2
            pad_left, pad_right = int(np.floor(pad_horz)), int(np.ceil(pad_horz))
            pad_top, pad_bot = 0, 0
        else:
            new_h, new_w = sh, sw
            pad_left = pad_right = pad_top = pad_bot = 0

        if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
            padColor = [padColor] * 3

        scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
        return cv2.copyMakeBorder(scaled, pad_top, pad_bot, pad_left, pad_right,
                                  borderType=cv2.BORDER_CONSTANT, value=padColor)

    def _removeImgPadding(self, img):
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    def saveResults(self, imageWebcam, imageCanvasNot, image):
        cv2.imwrite("signature.png", imageCanvasNot)
        cv2.imwrite("webcam.png",    imageWebcam)
        if self.pdfMode:
            imageNoPadding = self._removeImgPadding(image)
            cv2.imwrite("finalResult.png", imageNoPadding)
            imgPIL = Image.fromarray(imageNoPadding)
            imgPIL.save("signature_on_pdf.pdf", "PDF", resolution=100.0, save_all=True)
        else:
            cv2.imwrite("finalResult.png", image)
        print("Saved: signature.png, webcam.png, finalResult.png")