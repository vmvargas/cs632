""" This code demostrates reading the training and validation data
produced by extract_cats_dogs.py

It can be run from the command line, with one or two arguments:

$ python train.py <train.npy> <validation.npy>

where train and validation are a .npy file with an identical format to those 
produced by extract_cats_dogs.py for training and validation.

You will need to have Pillow installed to display the images.

http://pillow.readthedocs.io/en/3.4.x/installation.html
"""

from __future__ import print_function
import numpy as np
from PIL import Image
import keras
import os
import sys
import time
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

#constants
DATA_SIZE = 1  #percentage of current dataset to be used (.01 means 1%)
CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0
NUM_CLASSES = 2 #dogs or cats
BATCH_SIZE = 32
EPOCHS = 20
SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
MODEL_NAME = 'keras_cifar10_trained_model.h5'

if len(sys.argv)>1 and sys.argv[1]:
    TRAIN_PATH = sys.argv[1]
else:
    TRAIN_PATH = "train.npy"
    
if len(sys.argv)>2 and sys.argv[2]:
    VAL_PATH = sys.argv[2]
else:
    VAL_PATH = "validation.npy"

# helper function to plot a history of model's accuracy and loss
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
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

# reducing our dataset for testing purposes
x_train = train_images[:int(len(train_images)*DATA_SIZE)]
y_train = train_labels[:int(len(train_labels)*DATA_SIZE)]

x_test = val_images[:int(len(val_images)*DATA_SIZE)]
y_test = val_labels[:int(len(val_labels)*DATA_SIZE)]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

#initializing a Keras NN model based on a linear stack of layers
model = Sequential()
#input shape
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

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# training the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#convert our data type to float32 and normalize our data values to the range [0, 1].
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

start = time.time()
model_info = model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose=1)
end = time.time()
print ("\nModel training time: %0.1fs\n" % (end - start))

# plot model history
plot_model_history(model_info)

# Evaluating the trained model
scores = model.evaluate(x_test, y_test, verbose=0)
print("\nTest Loss:  %.2f%%" % (scores[0]*100))
print("Test Accuracy: %.2f%%\n" % (scores[1]*100))

# Saving model and weights
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
model_path = os.path.join(SAVE_DIR, MODEL_NAME)
model.save(model_path)
print('\nSaved trained model at %s ' % model_path)