import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
# import tensorflow_datasets as tfds

from tensorflow.keras import layers

data_dir = pathlib.Path("./data/full-dataset/train/clean/")
image_count = sum(1 for _ in data_dir.rglob('*.png'))
print(f"Number of images found: {image_count}")

if image_count == 0:
    raise ImportError("No images found (mine)")

batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,
    batch_size=batch_size,
    label_mode=None)

IMG_HEIGHT = 80
IMG_WIDTH = 400

# Main augment layers !!! USE THIS
augment_layers = tf.keras.Sequential([
    layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
    layers.Rescaling(1./255),
    layers.RandomBrightness(factor=0.2, value_range=[0.0, 1.0]),
    layers.RandomContrast(factor=0.3),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomZoom(height_factor=0.2, width_factor=0.1, fill_mode='nearest'),
    layers.RandomRotation(factor=0.02, fill_mode='nearest'),
])

# To see examples:

def show_images(dataset, augmentation_layer, num_images=5):
    plt.figure(figsize=(10, 10))
    for images in dataset.take(1):
        augmented_images = augmentation_layer(images)
        augmented_images = tf.clip_by_value(augmented_images, 0.0, 1.0)
        for i in range(num_images):
            _ = plt.subplot(1, num_images, i + 1)
            plt.imshow(augmented_images[i].numpy())
            plt.axis("off")
    plt.show()

show_images(train_ds, augment_layers, 3)