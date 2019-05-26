import cv2, os
import numpy as np
from PIL import Image


def get_images_and_labels(dataset_path, face_detector):
    images_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if
                    not f.endswith(".sad") and not f.endswith(".txt")]
    images, labels = [], []

    for image_path in images_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        label = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        faces = face_detector.detectMultiScale(image)
        for x, y, w, h in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(label)
            cv2.imshow("Adding face to training set", image[y: y + h, x: x + w])
            cv2.waitKey(50)

    return images, labels


cascade_path = "venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

dataset_path = 'yalefaces'
images, labels = get_images_and_labels(dataset_path, face_detector)
