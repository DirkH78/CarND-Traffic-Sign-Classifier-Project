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
image_shape = X_train[0].shape
 
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
 
# histogram
plt.hist(y_train, n_classes)
plt.xlim(0, n_classes-1)
plt.show()
 
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
     
def normImage(img):
    return -0.5 + (grayscale(img)/255) #normalizing the image
 
X_train_norm=np.zeros((n_train, 32,32,1))
for i in range (n_train):
    X_train_norm[i,:,:,0]=normImage(X_train[i,:,:,:])
 
X_test_norm=np.zeros((n_test, 32,32,1))
for i in range (n_test):
    X_test_norm[i,:,:,0]=normImage(X_test[i,:,:,:])
     
X_valid_norm=np.zeros((n_validation, 32,32,1))
for i in range (n_validation):
    X_valid_norm[i,:,:,0]=normImage(X_valid[i,:,:,:])
 
#for i in range(0,n_train,1000):
#    print(y_train[i])
#    plt.imshow(X_train_norm[i,:,:,0], cmap='gray')
#    plt.show() #show some test pictures with labels
 
### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten
import tensorflow as tf
 
 
def LeNet(x):   
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
     
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x56.
    conv_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 56), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(56))
    conv   = tf.nn.conv2d(x, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b
 
    # TODO: Activation.
    conv   = tf.nn.relu(conv)
     
    # TODO: Pooling. Input = 28x28x56. Output = 14x14x56.
    conv   = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
 
    # TODO: Layer 2: Convolutional. Output = 10x10x112.
    conv_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 56, 112), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(112))
    conv   = tf.nn.conv2d(conv, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b
     
    # TODO: Activation.
    conv   = tf.nn.relu(conv)
     
    # TODO: Pooling. Input = 10x10x112. Output = 5x5x112.
    conv   = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
 
    # TODO: Layer 3: Fully Connected. Input = 5x5x112. Output = 2x2x224.
    conv_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 112, 224), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(224))
    conv   = tf.nn.conv2d(conv, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b
     
    # TODO: Activation.
    conv   = tf.nn.relu(conv)
     
    # Dropout
    conv = tf.nn.dropout(conv, keep_prob)
     
    # TODO: Flatten. Input = 2x2x224. Output = 896.
    conv   = flatten(conv)
 
    # TODO: Layer 4: Fully Connected. Input = 896. Output = 224.
    conv_W = tf.Variable(tf.truncated_normal(shape=(896, 224), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(224))
    conv   = tf.matmul(conv, conv_W) + conv_b
     
    # TODO: Activation.
    conv   = tf.nn.relu(conv)
     
    # Dropout
    conv = tf.nn.dropout(conv, keep_prob)
 
    # TODO: Layer 5: Fully Connected. Input = 224. Output = 43.
    conv_W = tf.Variable(tf.truncated_normal(shape=(224, 43), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(43))
    logits   = tf.matmul(conv, conv_W) + conv_b
     
    return logits
 
#define placeholders
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
 
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
rate1 = 0.0005 #start value of rate
rate2 = 0.00001 #target value of rate
EPOCHS = 20
BATCH_SIZE = 128
KP = 0.4 #start value for keep_prob
 
rate = tf.placeholder(tf.float32) # dynamic rate
keep_prob = tf.placeholder(tf.float32) # probability to keep units
 
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
        accuracy = sess.run(accuracy_operation, feed_dict={x: X_data, y: y_data, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
     
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_norm, y_train = shuffle(X_train_norm, y_train)
        for offset in range(0, n_train, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_norm[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: KP+((1-KP)/(EPOCHS))*i, rate: rate1+((rate2-rate1)/(EPOCHS))*i})
                      
        validation_accuracy = evaluate(X_valid_norm, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Keep Prob = {:.4f}".format(KP+((1-KP)/(EPOCHS))*i))
        print("Rate = {:.6f}".format(rate1+((rate2-rate1)/(EPOCHS))*i))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
         
    saver.save(sess, './model.ckpt')
    print("Model saved")
 
with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')
    test_accuracy = evaluate(X_test_norm, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
 
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os
import cv2
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean
filelist=os.listdir("newExamples/")
filelist
noOfPic=len(filelist)
newtestx=np.zeros((noOfPic, 32,32,1))
for i in range (noOfPic):
    image = grayscale(mpimg.imread('newExamples/'+filelist[i])) #load all pictures in folder
    image_rescaled = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA) #resize to 32,32
    newtestx[i,:,:,0]=-0.5 + (image_rescaled/255) #nomalize
    plt.imshow(newtestx[i,:,:,0], cmap='gray')
    plt.show()
     
newtesty=np.array([1,12,13,14,17,38]) #labels for new set
 
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')
    test_accuracy = evaluate(newtestx, newtesty)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
 
### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
print("Test Accuracy [%] = {:.3f}".format(test_accuracy*100))
 
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
### Feel free to use as many code cells as needed.
softmaxOut=tf.nn.softmax(logits) #not necessary but quite pretty
with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')
    out=sess.run(softmaxOut, feed_dict={x: newtestx, keep_prob: 1})
    out=sess.run(tf.nn.top_k(out, k=5))
     
print(newtesty)
print(out)