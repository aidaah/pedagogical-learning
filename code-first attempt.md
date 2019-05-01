# pedagogical-learning
how children learn

```ruby
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from random import randint


# Preparing the dataset.
# Setup train and test splits.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Making a copy before flattening for the next code-segment which displays images (not used at the moment).
x_train_drawing = x_train

# Before the reshaping, x_train/test has the following shape (60000, 28, 28). This means 60k images of 28*28 pixels. 
# After the reshaping, it has the shape (60000, 784). x_train.shape[0] is the number of rows=60000, kept fixed, so it
# is only reshaping the 20*20 part.
image_size = 784 # 28 x 28
x_train = x_train.reshape(x_train.shape[0], image_size) 
x_test = x_test.reshape(x_test.shape[0], image_size)

# Convert class vectors to binary class matrices (one-hot encoding).
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```