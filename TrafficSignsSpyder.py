# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:45:34 2018

@author: Kunde
"""

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


import numpy as np
#print(len(np.unique(y_train)))
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = [28,28]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline
#for i in range(0,n_train,1000):
#    plt.imshow(X_train[i,:,:,:])
#    plt.show()


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import cv2
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
X_train_norm=np.zeros((n_train, 32,32,1))
for i in range (n_train):
    X_train_norm[i,:,:,0]=grayscale(X_train[i,:,:,:])
    X_train_norm[i,:,:,0]=(X_train_norm[i,:,:,0]-128)/128

X_test_norm=np.zeros((n_test, 32,32,1))
for i in range (n_test):
    X_test_norm[i,:,:,0]=grayscale(X_test[i,:,:,:])
    X_test_norm[i,:,:,0]=(X_test_norm[i,:,:,0]-128)/128
    
X_valid_norm=np.zeros((n_validation, 32,32,1))
for i in range (n_validation):
    X_valid_norm[i,:,:,0]=grayscale(X_valid[i,:,:,:])
    X_valid_norm[i,:,:,0]=(X_valid_norm[i,:,:,0]-128)/128
#plt.imshow(X_train_norm[0,:,:], cmap='gray')
#X_train_norm[0,:,:]
    
 ### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten
import tensorflow as tf

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(6))
    conv   = tf.nn.conv2d(x, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b

    # TODO: Activation.
    conv   = tf.nn.relu(conv)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv   = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(16))
    conv   = tf.nn.conv2d(conv, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b
    
    # TODO: Activation.
    conv   = tf.nn.relu(conv)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv   = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    conv   = flatten(conv)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    conv_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(120))
    conv   = tf.matmul(conv, conv_W) + conv_b
    
    # TODO: Activation.
    conv   = tf.nn.relu(conv)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    conv_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(84))
    conv   = tf.matmul(conv, conv_W) + conv_b
    
    # TODO: Activation.
    conv   = tf.nn.relu(conv)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    conv_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(43))
    logits   = tf.matmul(conv, conv_W) + conv_b
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)   

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
rate = 0.001
EPOCHS = 1 #10
BATCH_SIZE = 128

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: X_data, y: y_data})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_norm, y_train = shuffle(X_train_norm, y_train)
        for offset in range(0, n_train, BATCH_SIZE):
            #print("Offset progress ...".format(offset/n_train))
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_norm[offset:end], y_train[offset:end]
            #print(np.shape(batch_x))
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid_norm, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


    