from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import pandas as pd

# Initializing the CNN
classifier = Sequential()

# step - 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(50, 50, 3), activation='relu'))

# step -2 -- Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step - 3 -- Flattening
classifier.add(Flatten())

# Step -4 Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# part - 2 -- Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/train', # path/to/data/
    target_size=(50, 50),
    batch_size=32,
    class_mode='binary'
)
test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(50, 50),
    batch_size=32,
    class_mode='binary'
)

classifier.fit_generator(
    training_set,
    samples_per_epoch=400,
    nb_epoch=25,
    validation_data=test_set,
    nb_val_samples=100
)

#save model
classifier.save('dino-model.h5')
