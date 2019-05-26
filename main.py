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

    return images, labels


cascade_path = "venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

dataset_path = 'yalefaces'
images, labels = get_images_and_labels(dataset_path, face_detector)

face_recognizer = cv2.face_LBPHFaceRecognizer.create()
face_recognizer.train(images, np.array(labels))

test_images_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".sad")]
correct_predictions = 0
for test_image_path in test_images_paths:
    test_image = np.array(Image.open(test_image_path).convert('L'), 'uint8')
    faces = face_detector.detectMultiScale(test_image)
    for x, y, w, h in faces:
        label_predicted, conf = face_recognizer.predict(test_image[y: y + h, x: x + w])
        label_actual = int(os.path.split(test_image_path)[1].split(".")[0].replace("subject", ""))
        if label_actual == label_predicted:
            correct_predictions += 1
            print("{} Recognized correctly with confidence {}".format(label_actual, conf))
        else:
            print("{} Recognized incorrectly as {}".format(label_actual, label_predicted))

print("Accuracy: {}".format(100 * correct_predictions / len(test_images_paths)))
