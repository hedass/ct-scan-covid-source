import tensorflow as tf
from custom_objects import UpsampleLike
import cv2
import numpy as np
import os
import json
from numba import cuda


class Classification():
    def __init__(self, src, dst):
        self.model_path = "./models/ResNet50V2-FPN-fold15-16-0.8997.hdf5"
        self.src = src
        self.dst = dst
        self.classes = ["Normal", "COVID", "Pneumonia"]
        self.n_classes = len(self.classes)

    def calculate_with_argmax(self, prediction):
        return np.argmax(prediction)

    def calculate_with_threshold(self, prediction, threshold=0.1):
        total = np.float(np.sum(prediction))
        if(prediction[1] / total >= threshold):
            return 1
        elif(prediction[2] / total >= threshold):
            return 2
        else:
            return 0

    def predict_from_folder(self, model, foldername):
        prediction = np.array([0 for _ in range(self.n_classes)])
        files = os.listdir(foldername)
        for file in files:
            img_name = os.path.join(foldername, file)
            img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            pred = np.argmax(model.predict(np.expand_dims(img, axis=0))[0])
            prediction[pred] += 1
        result = self.calculate_with_argmax(prediction)
        prediction = prediction / sum(prediction)
        prediction = {
            "Normal": float(prediction[0]), "COVID": float(prediction[1]), "Pneumonia": float(prediction[2])}
        return {"predictedClass": self.classes[result], "probability": prediction}

    def run(self):
        model = tf.keras.models.load_model(self.model_path, custom_objects={
                                           "UpsampleLike": UpsampleLike}, compile=False)
        pred = self.predict_from_folder(model, self.src)
        with open(os.path.join(self.dst, "classification.json"), 'w') as f:
            json.dump(pred, f)
        del model
