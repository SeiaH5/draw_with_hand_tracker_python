import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import math
from tkinter import filedialog
import os
from pdf2image import convert_from_path
from PIL import Image
import pathlib
import urllib.request

# ── Download model if not present ──────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def pointDistance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def draw(canvas, position, last_position, thickness, colour):
    cv2.circle(canvas, position, thickness // 2, colour, cv2.FILLED)
    if last_position[0] is not None and last_position[1] is not None:
        cv2.line(canvas, last_position, position, colour, thickness)

def checkFingerVerticalUp(finger, threshold):
    tip = landmark_pts[fingerTipIDs[finger]]
    mcp = landmark_pts[fingerTipIDs[finger] - 2]
    return math.fabs(tip[0] - mcp[0]) <= threshold and tip[1] < mcp[1]

def checkFingersHorizontal():
    for i in range(1, 5):
        tip = landmark_pts[fingerTipIDs[i]]
        pip = landmark_pts[fingerTipIDs[i] - 2]
        if math.fabs(tip[1] - pip[1]) >= 20:
            return False
    return True

def posOnScreen(p):
    return 0 <= p[0] <= vid_width and 0 <= p[1] <= vid_height

def checkFingersDownCondition(req):
    return all(fingersUp[i] == 0 for i in req)

def checkThumbsUp():
    return (
        all(posOnScreen(landmark_pts[f]) and posOnScreen(landmark_pts[f - 2]) for f in fingerTipIDs)
        and checkFingerVerticalUp(0, 20)
        and fingersUp[0]
        and checkFingersDownCondition([1, 2, 3, 4])
        and checkFingersHorizontal()
    )

def checkFingersUp(fingersUpList):
    palm = landmark_pts[0]
    for i in range(5):
        tip = landmark_pts[fingerTipIDs[i]]
        pip = landmark_pts[fingerTipIDs[i] - 1]
        dip = landmark_pts[fingerTipIDs[i] - 2]
        if i == 0:
            if pointDistance(tip, palm) > pointDistance(pip, palm):
                fingersUpList[0] = 1
        else:
            if pointDistance(tip, palm) > pointDistance(dip, palm):
                fingersUpList[i] = 1

def saveResultsPNG():
    cv2.imwrite("signature.png", imageCanvasNot)
    cv2.imwrite("webcam.png", imageWebcam)
    if pdfMode:
        imageNoPadding = removeImgPadding(image)
        cv2.imwrite("finalResult.png", imageNoPadding)
        imgPIL = Image.fromarray(imageNoPadding)
        imgPIL.save("signature_on_pdf.pdf", "PDF", resolution=100.0, save_all=True)
    else:
        cv2.imwrite("finalResult.png", image)
    print("Saved: signature.png, webcam.png, finalResult.png")

def removeImgPadding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def resizeAndPad(img, size, padColor=0):
    h, w    = img.shape[:2]
    sh, sw  = size
    interp  = cv2.INTER_AREA if (h > sh or w > sw) else cv2.INTER_CUBIC
    aspect  = w / h

    if aspect > 1:
        new_w = sw
        new_h = int(np.round(new_w / aspect))
        pad_vert  = (sh - new_h) / 2
        pad_top, pad_bot = int(np.floor(pad_vert)), int(np.ceil(pad_vert))
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = int(np.round(new_h * aspect))
        pad_horz  = (sw - new_w) / 2
        pad_left, pad_right = int(np.floor(pad_horz)), int(np.ceil(pad_horz))
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left = pad_right = pad_top = pad_bot = 0

    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled = cv2.copyMakeBorder(scaled, pad_top, pad_bot, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=padColor)
    return scaled

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks_on_image(image, pts):
    """Draw hand landmarks using plain OpenCV - no mediapipe.framework needed."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
    for i, pt in enumerate(pts):
        colour = (0, 0, 255) if i in fingerTipIDs else (255, 0, 0)
        cv2.circle(image, tuple(pt), 5, colour, cv2.FILLED)

# ── MediaPipe Tasks setup ──────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ── Constants ──────────────────────────────────────────────────────────────────
vid_width, vid_height = 1280, 720

video = cv2.VideoCapture(0)
video.set(3, vid_width)
video.set(4, vid_height)

imageCanvas = np.zeros((vid_height, vid_width, 3), np.uint8)

# BGR colours
colours = [
    (0, 0, 0),       # 0 black
    (255, 0, 0),     # 1 blue
    (0, 255, 0),     # 2 green
    (0, 0, 255),     # 3 red
    (0, 255, 255),   # 4 yellow
    (0, 165, 255),   # 5 orange
    (128, 0, 128),   # 6 purple
    (203, 192, 255), # 7 pink
    (255, 255, 0),   # 8 cyan
    (255, 255, 255), # 9 white
]
curr_colour = 0

fingerTipIDs = [4, 8, 12, 16, 20]
pen_thickness   = 8
erase_thickness = 30
index_lx, index_ly = None, None

canWrite  = False
eraseMode = False
pdfMode   = False
imgMode   = False
running   = True

imagePDF   = None
imgNPRGB   = None

win_name = "Output"

print("Controls:")
print("  W - toggle draw mode")
print("  E - toggle erase mode")
print("  C - clear canvas")
print("  O - open PDF or image file")
print("  R - reset everything")
print("  , / . - decrease / increase brush size")
print("  0-9 - change colour")
print("  S or thumbs-up gesture - save and quit")
print("  Q - quit without saving")

# ── Main loop ──────────────────────────────────────────────────────────────────
while running:
    success, imageWebcam = video.read()
    if not success:
        break

    imageWebcam = cv2.flip(imageWebcam, 1)

    if not pdfMode and not imgMode:
        image = imageWebcam.copy()
    elif pdfMode:
        image = resizeAndPad(imagePDF, (vid_height, vid_width))
    else:
        image = resizeAndPad(imgNPRGB, (vid_height, vid_width))

    image_h, image_w = image.shape[:2]

    imageCanvasNot = cv2.bitwise_not(imageCanvas)
    imageCanvasNot[np.all(imageCanvasNot == (0, 0, 0), axis=-1)] = colours[curr_colour]
    image = np.where(imageCanvas == (0, 0, 0), image, imageCanvasNot)

    # ── Hand detection with new Tasks API ─────────────────────────────────────
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(imageWebcam, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    landmark_pts = []
    fingersUp    = [0] * 5

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        for lm in hand_landmarks:
            cx = int(lm.x * image_w)
            cy = int(lm.y * image_h)
            landmark_pts.append([cx, cy])

        checkFingersUp(fingersUp)

        index_cx = landmark_pts[fingerTipIDs[1]][0]
        index_cy = landmark_pts[fingerTipIDs[1]][1]

        drawMode = [2, 3, 4]

        if canWrite and fingersUp[1] and checkFingersDownCondition(drawMode):
            draw(imageCanvas, (index_cx, index_cy), (index_lx, index_ly),
                 pen_thickness, (255, 255, 255))
            index_lx, index_ly = index_cx, index_cy
        elif eraseMode and fingersUp[1] and checkFingersDownCondition(drawMode):
            draw(imageCanvas, (index_cx, index_cy), (index_lx, index_ly),
                 erase_thickness, (0, 0, 0))
            index_lx, index_ly = index_cx, index_cy
        else:
            index_lx, index_ly = None, None

        # Thumbs up → save and quit
        if not canWrite and checkThumbsUp():
            saveResultsPNG()
            running = False

        # Pinky up → clear canvas
        if not canWrite and checkFingersDownCondition([1, 2, 3]) and checkFingerVerticalUp(4, 50):
            imageCanvas = np.zeros((vid_height, vid_width, 3), np.uint8)

        # Draw fingertip cursor
        cv2.circle(image,
                   (landmark_pts[fingerTipIDs[1]][0], landmark_pts[fingerTipIDs[1]][1]),
                   erase_thickness // 2, (0, 255, 255), cv2.FILLED)

        # Draw landmarks
        draw_landmarks_on_image(image, landmark_pts)

    # ── HUD ───────────────────────────────────────────────────────────────────
    mode_text = "DRAW" if canWrite else ("ERASE" if eraseMode else "IDLE")
    cv2.putText(image, f"Mode: {mode_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    thickness_val = pen_thickness if canWrite else erase_thickness
    cv2.putText(image, f"Size: {thickness_val}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(image, (10, 75), (40, 105), colours[curr_colour], cv2.FILLED)

    cv2.imshow(win_name, image)
    cv2.moveWindow(win_name, 0, 0)

    # ── Key handling ──────────────────────────────────────────────────────────
    keys = cv2.waitKey(1) & 0xFF

    if keys == ord('q'):
        running = False
    elif keys == ord('s'):
        saveResultsPNG()
        running = False
    elif keys == ord('w'):
        canWrite = not canWrite
        if canWrite:
            eraseMode = False
    elif keys == ord('e'):
        eraseMode = not eraseMode
        if eraseMode:
            canWrite = False
    elif keys == ord('c'):
        imageCanvas = np.zeros((vid_height, vid_width, 3), np.uint8)
    elif keys == ord('r'):
        index_lx, index_ly = None, None
        canWrite  = False
        eraseMode = False
        pdfMode   = False
        imgMode   = False
        running   = True
        imageCanvas     = np.zeros((vid_height, vid_width, 3), np.uint8)
        pen_thickness   = 8
        erase_thickness = 30
        curr_colour     = 0
    elif keys == ord(','):
        if canWrite and pen_thickness - 6 >= 4:
            pen_thickness -= 6
        elif eraseMode and erase_thickness - 6 >= 4:
            erase_thickness -= 6
    elif keys == ord('.'):
        if canWrite:
            pen_thickness += 6
        elif eraseMode:
            erase_thickness += 6
    elif ord('0') <= keys <= ord('9'):
        curr_colour = int(chr(keys))
    elif keys == ord('o'):
        file = filedialog.askopenfile(
            mode='r',
            filetypes=[
                ("PDF and images", "*.pdf *.png *.jpg *.jpeg *.gif *.bmp *.ico"),
                ("PDF file", "*.pdf"),
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")
            ]
        )
        if file:
            filepath = os.path.abspath(file.name)
            suffix   = pathlib.Path(file.name).suffix.lower()
            if suffix == ".pdf":
                pages    = convert_from_path(filepath)
                imagePDF = np.array(pages[0])
                pdfMode  = True
                imgMode  = False
            else:
                imgPic   = Image.open(filepath)
                imgNP    = np.asarray(imgPic)
                imgNPRGB = cv2.cvtColor(imgNP, cv2.COLOR_RGB2BGR)
                imgMode  = True
                pdfMode  = False

video.release()
cv2.destroyAllWindows()