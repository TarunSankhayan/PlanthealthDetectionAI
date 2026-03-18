print("Prediction script started")

import numpy as np
import cv2
from tensorflow.keras.models import load_model

# load trained model
model = load_model("../models/plant_disease_model.h5")

# class labels (must match training output)
labels = [
    "Tomato Early blight",
    "Tomato Late blight",
    "Tomato healthy"
]

def predict_disease(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)

    disease = labels[index]
    confidence = round(float(prediction[0][index])*100, 2)

    return disease, confidence


if __name__ == "__main__":

    disease, confidence = predict_disease("test_leaf.jpg")

    print("Prediction:", disease)
    print("Confidence:", confidence)