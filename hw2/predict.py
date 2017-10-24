""" This code demonstrates reading the test data and writing 
predictions to an output file.

It can be run from the command line, with one argument:

$ python predict.py <test_file.npy>

where test_file is a .npy file with an identical format to those 
produced by extract_cats_dogs.py for training and validation.

This script will create an output file in the same directory 
where it's run, called "predictions.txt".

"""

import sys
import numpy as np
import random
import os
import keras
from keras.models import load_model

#constants
CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0
# This file will be created if it does not exist
# and overwritten if it does
OUT_FILE = "predictions.txt"
LOAD_DIR = os.path.join(os.getcwd(), 'saved_models/keras_cifar10_trained_model.h5')
DATA_SIZE = 1  #percentage of current dataset to be used (.01 means 1%)
BATCH_SIZE = 32

if len(sys.argv)>1 and sys.argv[1]:
    TEST_FILE = sys.argv[1]
else:
    TEST_FILE = "validation.npy"

data = np.load(TEST_FILE).item()
images = data["images"]

# the testing data also contains a unique id
# for each testing image
if "ids" in data:
    ids = data["ids"]
else:
    #if it's not contained, a sequence is used
    ids = list(range(0,len(images)))
    ids = np.array(ids)

# reducing our dataset for testing purposes.
x_train = images[:int(len(images)*DATA_SIZE)]

model = load_model(LOAD_DIR)

print("\nPredicting...")
y_pred = model.predict_classes(x_train, verbose=1)

# making a prediction on each image
# and writing output to disk
out = open(OUT_FILE, "w")
for i, image in enumerate(images):
  line = str(ids[i]) + " " + str(y_pred[i]) + "\n"
  out.write(line)
out.close()
print("\nPredictions were written to the file: ",OUT_FILE)