'''
this script is meant to reorganize the dataset train directory so it extracts
a validation dataset from it
'''
import os
import shutil
import random

# set paths
base_dir = 'dset'
val_dir= os.path.join(base_dir, 'val')
train_dir= os.path.join(base_dir, 'train')
cat_dir = os.path.join(train_dir, 'cat')
not_cat_dir = os.path.join(train_dir, 'not_cat')

# make validation directories
os.makedirs(val_dir, exist_ok=True)
val_cat_dir = os.path.join(val_dir, 'cat')
os.makedirs(val_cat_dir, exist_ok=True)
val_not_cat_dir = os.path.join(val_dir, 'not_cat')
os.makedirs(val_not_cat_dir, exist_ok=True)

# Function to move all files from source to destination directory
def move_all_files(src_dir, dest_dir):
    for filename in os.listdir(src_dir):
        shutil.move(os.path.join(src_dir, filename), dest_dir)

# Move any existing files in the validation directories back to the training directories
move_all_files(val_cat_dir, cat_dir)
move_all_files(val_not_cat_dir, not_cat_dir)

# shuffle and extract images
def shuffle_and_extract(src_dir, dest_dir, num_images):
    files = os.listdir(src_dir)
    random.shuffle(files)
    for i in range(min(num_images, len(files))):
        shutil.move(os.path.join(src_dir, files[i]), dest_dir)

# extract 500 images from each directory
shuffle_and_extract(cat_dir, val_cat_dir, 800)
shuffle_and_extract(not_cat_dir, val_not_cat_dir, 800)
