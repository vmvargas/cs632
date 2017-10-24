# A Convolutional Neural Network about Cats and Dogs 


Deep CNN on the CIFAR10 small images dataset, using only images of cats and dogs.


## Getting Started

Clone this repo and run: 

```
$ pip install -r requirements.txt
$ curl -O http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
$ tar xvf cifar-10-python.tar.gz
```

## Deployment

1. ```$ python extract_cats_dogs.py``` creates training and validation data from CIFAR10 (cats and dogs only).


2. ```$ python train.py <train.npy> <validation.npy> ``` trains and​ evaluates​ a neural​ network​ that ​ classifies​ the training​ set. This​ file​ reads​ a​ training​ dataset​ ( 1st command line argument) and a ```validation.npy``` ( 1st command line argument) or uses the output of ```extract_cats_dogs.py``` if no args are passed. Then, it trains a model,​ evaluates​ its​ accuracy and finally save​ a checkpoint​ to​ disk.


3. ```$ python predict.py <test.npy>``` loads the model saved by ```train.py```, reads the testing dataset ```<test.npy>``` (or ```validation.npy``` if no args are passed) to make predictions for every image. These predictions are saved in a filled called ```predictions.txt```. 


## Built With

* Keras
* Tensorflow
* Python 3.6


