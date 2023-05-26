from PIL import Image
import os

def find_bad_images(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')): # add any other image types if necessary
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # verify that it is, in fact an image
                except (IOError, SyntaxError) as e:
                    print('Bad file:', file_path)  # print out the names of corrupt or bad files

base_dir = 'dset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# search in training and validation directories
find_bad_images(train_dir)
find_bad_images(validation_dir)

