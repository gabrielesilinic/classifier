import tensorflow as tf
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'dset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

batch_size = 32

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)


val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(200, 200),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary')

# Calculate steps per epoch
train_steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,  # Adjusted
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_steps,  # Adjusted
    verbose=2)

model.save('cat_or_not_cat_model.h5')

metadata = {
    'class_indices': train_generator.class_indices,
    'input_shape': (200, 200, 1),
    'epochs': 12,
    'optimizer': 'RMSprop',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f)
