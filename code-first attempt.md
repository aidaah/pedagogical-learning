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
``` ruby
# Symbols
X = tf.placeholder("float", shape=[None, 784])
ylabel = tf.placeholder("float", shape=[None, 10])
a = tf.placeholder("float", shape=None)

# Weight initializations
w_1 = tf.Variable(tf.random_normal((784, 32), stddev=0.1))
b_1 = tf.Variable(tf.random_normal((1, 32), stddev=0.1))
w_2 = tf.Variable(tf.random_normal((32, 10), stddev=0.1))
b_2 = tf.Variable(tf.random_normal((1, 10), stddev=0.1))

h = tf.nn.sigmoid(tf.matmul(X, w_1) + b_1)
ypred = tf.matmul(h, w_2) + b_2

# Backward propagation
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ylabel, logits=ypred))
loss2 = tf.reduce_mean((ylabel-ypred)**2)
loss3 = (1-a)*loss2+a*loss1
optimizer = tf.train.AdamOptimizer()
updates = optimizer.minimize(loss3)
```
```ruby
print(tf.trainable_variables())
grad = [a for a, b in optimizer.compute_gradients(loss3, var_list=tf.trainable_variables())]
print(grad)
```

```ruby
# Run SGD
sess = tf.Session()
sess.run(tf.global_variables_initializer())

aa = 0
for epoch in range(10):
    # Train with each example
    for i in range(len(x_train)):
        sess.run(updates, feed_dict={X: x_train, ylabel: y_train, a: aa})
        
        if i % 20 == 0:
            y_train_pred = sess.run(ypred, feed_dict={X: x_train})
            y_test_pred = sess.run(ypred, feed_dict={X: x_test})
            
            train_accuracy = np.mean(np.argmax(y_train_pred, axis=1) == np.argmax(y_train, axis=1))
            test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))
        
            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
            
            ## To compute the gradient of the loss3 with w1,w2,b1,b2, which gives a list [dw1, dw2, db1, db2]
            ## Note: dw1, dw2, db1 and db2 are all matrices
            all_grad_value = [sess.run(grad[i], feed_dict={X: x_train, ylabel: y_train, a: aa})
                              for i in range(len(grad))]
            #print("all_grad_value")
            #print(all_grad_value)
            
            ## To compute the norm for each of dw1, dw2, db1, db2, which gives 4 numbers,
            ## and add them together
            grad_norm = 0
            for i in range(len(grad)):
                each_grad_norm = np.sum(abs(all_grad_value[i]))
                grad_norm = grad_norm + each_grad_norm
            print("grad_norm : ", grad_norm)
            
            if grad_norm < 1 and aa <= 1:
                aa += 0.01
                print("update a: ", aa)

sess.close()
```
