""" This code demostrates reading the training and validation data
produced by extract_cats_dogs.py

You will need to have Pillow installed to display the images.

http://pillow.readthedocs.io/en/3.4.x/installation.html
"""

from __future__ import print_function
import numpy as np
from PIL import Image
import keras
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

#constants
TRAIN_PATH = "train.npy"
VAL_PATH = "validation.npy"
DATA_SIZE = .01  #percentage of current dataset to be used (.01 means 1%)
CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0
NUM_CLASSES = 2 #dogs or cats
BATCH_SIZE = 32
EPOCHS = 10
SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
MODEL_NAME = 'keras_cifar10_trained_model.h5'

def load(npy_file):
  data = np.load(npy_file).item()
  return data['images'], data['labels']

train_images, train_labels = load(TRAIN_PATH)
val_images, val_labels = load(VAL_PATH)

# Make sure the images look correct
#i = 0
#image = train_images[i]
#label = train_labels[i]
#if label == CAT_OUTPUT_LABEL: 
#  print ("Cat!")
#else:
#  print ("Dog!")
#
#im = Image.fromarray(image)
#im.show()

# reducing our dataset for testing purposes.
x_train = train_images[:int(len(train_images)*DATA_SIZE)]
y_train = train_labels[:int(len(train_labels)*DATA_SIZE)]

x_test = val_images[:int(len(val_images)*DATA_SIZE)]
y_test = val_labels[:int(len(val_labels)*DATA_SIZE)]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

#initializing a Keras NN model based on a linear stack of layers
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

# initializing the RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# training the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose=0)

# Evaluating the trained model
scores = model.evaluate(x_test, y_test, verbose=0)
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1])

y_pred = model.predict(x_train, BATCH_SIZE, verbose=0)
print('\n y_pred:\n', y_pred)
print('\n y_train:\n', y_train)
#y_pred = keras.utils.to_categorical(y_pred, NUM_CLASSES)


# Saving model and weights
#if not os.path.isdir(SAVE_DIR):
#    os.makedirs(SAVE_DIR)
#model_path = os.path.join(SAVE_DIR, MODEL_NAME)
#model.save(model_path)
#print('\nSaved trained model at %s ' % model_path)