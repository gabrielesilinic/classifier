'''
this script is supposed to help to identify
all images the model does not have high confidence
that it is a cat or it is a cat depending on the category
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
import json
import argparse
import shutil
from tensorflow.keras.preprocessing import image
from tkinter import Tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from tkinter import messagebox
from tqdm import tqdm  # Import tqdm here

exit_button_clicked = False
def close_window(root):
        global exit_button_clicked
        exit_button_clicked = True
        root.quit()
        exit(0)

# Parsing CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specify the directory to check images")
parser.add_argument("--iscat", default='false', choices=['true', 'false'])
parser.add_argument("--bias", type=float, default=0.5)
parser.add_argument("--max", type=int, default=float('inf'))
parser.add_argument("--del", dest='delete', default='false', choices=['true', 'false'])
parser.add_argument("--copy", default=None)
parser.add_argument("--select", action='store_true')
args = parser.parse_args()

# Loading model and its metadata
model = tf.keras.models.load_model('alt_models/cat_or_not_cat_model_epoch04fromargv.h5')
with open('alt_models/metadata_retrain2.json', 'r') as f:
    metadata = json.loads(f.read())

# Adjust class indices based on iscat argument
if args.iscat == 'false':
    target_class = 1
else:
    target_class = 0

# Initializing counters
count = 0

# Get list of images
images = [f for f in os.listdir(args.dir) if f.endswith(".jpg") or f.endswith(".png")]

# Check images in the directory
for filename in tqdm(images, desc="Analyzing images"):  # Wrap the loop in tqdm to create a progress bar
    # Load image
    img_path = os.path.join(args.dir, filename)
    img = image.load_img(img_path, color_mode='grayscale', target_size=(metadata['input_shape'][0], metadata['input_shape'][1]))
    img_tensor = image.img_to_array(img)
    img_tensor = img_tensor / 255.
    img_tensor = img_tensor.reshape(1, metadata['input_shape'][0], metadata['input_shape'][1], metadata['input_shape'][2])
    # Predict
    prediction = model.predict(img_tensor,verbose=0)
    predicted_class = int(prediction > args.bias)

    # Check if the image matches the target
    # GUI
    if args.select and predicted_class == target_class:
        # Show image
        count += 1
        root = Tk()
        img = Image.open(img_path)
        img.thumbnail((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        panel = Label(root, image=photo)
        panel.pack()
        exit_button = Button(root, text="Exit", command=lambda: close_window(root))
        exit_button.pack()
        v = messagebox.askyesno("Select confidence: {}".format(prediction), "Is this a cat?" if target_class == 0 else "Is this not a cat?")
        root.destroy()
        if count >= args.max:
            break
        if not v:
            continue
    if predicted_class == target_class:
        count += 1
        print("\n"+filename)
    # Copy image
    if args.copy is not None:
        shutil.copy(img_path, args.copy)
    # Delete image
    if args.delete == 'true':
        os.remove(img_path)
        # Max images
    if count >= args.max:
        break

sys.exit(0)
