import urllib.request
import os
from gesture_ink import GestureInk

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

if __name__ == "__main__":
    app = GestureInk()
    app.run()