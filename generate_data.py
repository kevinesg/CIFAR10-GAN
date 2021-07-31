import GAN
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import os

BATCH_SIZE = 128
EPOCHS = 100
CODINGS_SIZE = 100
CLASS_SIZE = 10000  # This is the number of new data points per class (10 classes)

# Load labels from the csv file
labels = np.array(pd.read_csv('dataset/trainLabels.csv')['label'])
# Encode the labels into numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels).reshape(labels.shape[0], 1)

print('[INFO] Processing training images...')
# Initialize the list of training images
imgs_list = []
# Loop over each training image
for image in os.listdir('dataset/train/'):
    image_path = 'dataset/train/' + image
    cv2_img = cv2.imread(image_path)
    # Rescale values from [0, 255] to [-1, 1]
    # (this is due to tanh activation of the generator)
    img_array = np.array(cv2_img) / 255 * 2 - 1
    imgs_list.append(img_array)
imgs_array = np.array(imgs_list)

# Loop over each of the 10 classes
for class_num in range(10):
    # Get the indices of the labels we're concerned with
    i = labels[:, 0]==class_num
    # Apply masking to the images array
    data = tf.constant(imgs_array.copy()[i])
    # Prepare data for training
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0])
    dataset = dataset.batch(int(BATCH_SIZE / 2), drop_remainder=True).prefetch(1)

    # Instantiate the GAN layers
    generator = GAN.generator(100)
    discriminator = GAN.discriminator()

    gan = keras.models.Sequential([
        generator,
        discriminator
    ])

    # Compile the discriminator and whole GAN
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    )

    # Freeze discriminator layers so that only generator will be trained
    discriminator.trainable = False
    gan.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    )

    # Train the GAN
    print(f'[INFO] Training GAN for class #{class_num}...')
    GAN.train_gan(gan, dataset, BATCH_SIZE, CODINGS_SIZE, class_num, EPOCHS)

    # Generate synthetic data
    noise = np.random.uniform(size=(CLASS_SIZE, CODINGS_SIZE))
    # Rescale the generator output from [-1, 1] to [0, 1]
    gen_imgs = generator.predict(noise) / 2 + 0.5
    # Flatten each image 2D array
    gen_imgs = gen_imgs.reshape([gen_imgs.shape[0], 3072])
    # Generate a 1D array of labels
    labels_num = np.full((gen_imgs.shape[0], 1), class_num)
    new_data = np.hstack([labels_num, gen_imgs])
    new_data = pd.DataFrame(new_data)
    new_data.iloc[:, 0] = new_data.iloc[:, 0].astype(np.int8)
    # Save new dataset
    print('[INFO] Saving new dataset...')
    new_data.to_csv(f'dataset/new_data/{class_num}.csv', index=False)
    
print('[INFO] All done!')