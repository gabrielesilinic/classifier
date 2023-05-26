'''
this script just helps you to do predictions on an image you specify as a path
'''

import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

def load_and_preprocess_image(img_path):
    # Load image, grayscale, resize to match model's expected input shape
    img = image.load_img(img_path, target_size=(200, 200), color_mode='grayscale')
    # Convert image to array and normalize
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.  # normalize to [0,1] range
    return img_tensor

def predict_cat_or_not(img_path, model_path='alt_models/cat_or_not_cat_model_epoch04fromargv.h5'):
    # Load trained model
    model = load_model(model_path)
    # Load and preprocess input image
    img_tensor = load_and_preprocess_image(img_path)
    # Make prediction
    prediction = model.predict(img_tensor)
    return prediction

def main():
    img_path = sys.argv[1]
    prediction = predict_cat_or_not(img_path)
    if prediction[0][0] > 0.5:
        print(f"The image at {img_path} is predicted to not be a cat.")
    else:
        print(f"The image at {img_path} is predicted to be a cat.")
    print("not a cat confidence: "+str(prediction[0][0]))

if __name__ == "__main__":
    main()
