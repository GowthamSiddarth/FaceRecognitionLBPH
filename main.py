import cv2

cascade_path = "venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_recognizer = cv2.CascadeClassifier(cascade_path)
