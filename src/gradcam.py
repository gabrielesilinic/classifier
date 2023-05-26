'''
this script shows the CNN's activations on inference with a specific input
it may help you to improve the dataset or the architecture like i did
this script was mostly automatically generated using ChatGPT v4 so it may
or may not be accurate
'''
import sys
import argparse
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Generate a Grad-CAM heatmap for a given image.')
    parser.add_argument('image', type=str, help='Path to the input image.')
    parser.add_argument('--show', action='store_true', help='Show the resulting heatmap.')
    args = parser.parse_args()

    # load model and meta data
    model = load_model('alt_models/cat_or_not_cat_model_epoch04fromargv.h5')
    with open('alt_models/metadata_retrain2.json') as json_file:
        data = json.load(json_file)

    # get image
    img_path = args.image
    img = image.load_img(img_path, target_size=(200, 200), color_mode="grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.

    # predict the output
    preds = model.predict(x)

    # output the feature map of the last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        print("No convolutional layers found")
        sys.exit()
    
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, np.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (200, 200), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # Superimpose the CAM on original image
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (200, 200))

    # convert grayscale image to 3-channel image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(img_color)
    cam = 255 * cam / np.max(cam)

    cv2.imwrite("heatmap.jpg", np.uint8(cam))

    if args.show:
        plt.imshow(cv2.cvtColor(np.uint8(cam), cv2.COLOR_BGR2RGB))
        plt.show()

if __name__ == '__main__':
    main()
