---
title: "Image Classification"
layout: post
excerpt: "This script will classify images from the CIFAR-10 dataset"
tags: [Python, Udacity, Deep Learning, Neural network, Image classification, CIFAR-10, image prepocessing, backpropagation, tensorflow, Convolutional Neural Networks]
header:
  teaser: cifar2.jpg
categories: portfolio
link:
share: true
comments: false
date: 11/11/2017
---
## Summary
This project has been done while studying the Deep Learning Nanodegree at [Udacity](http://udacity.com/). There are some unit test functions to be passed before submitting this project. I did not include its code as it is not the main purpose of this script.

In this project, I'll classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset consists of airplanes, dogs, cats, and other objects. First I will preprocess the images, then train a convolutional neural network on all the samples. The images need to be normalized and the labels need to be one-hot encoded. I'll get to apply everything I learned and build a convolutional, max pooling, dropout, and fully connected layers.

## Getting the data

```python
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

# Use Floyd's cifar-10 dataset if present
floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)
```
All files found!

## Exploring the Data

The dataset is broken into batches to prevent your machine from running out of memory. The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
+ airplane
+ automobile
+ bird
+ cat
+ deer
+ dog
+ frog
+ horse
+ ship
+ truck

Understanding a dataset is part of making predictions on the data. I can play around with the code cell below by changing the `batch_id` and `sample_id`. The `batch_id` is the id for a batch (1-5). The `sample_id` is the id for a image and label pair in the batch.

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np

# Explore the dataset
batch_id = 4
sample_id = 10
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```
Stats of batch 4:
Samples: 10000
Label Counts: {0: 1003, 1: 963, 2: 1041, 3: 976, 4: 1004, 5: 1021, 6: 1004, 7: 981, 8: 1024, 9: 983}
First 20 Labels: [0, 6, 0, 2, 7, 2, 1, 2, 4, 1, 5, 6, 6, 3, 1, 3, 5, 5, 8, 1]

Example of Image 10:
Image - Min Value: 44 Max Value: 246
Image - Shape: (32, 32, 3)
Label - Label Id: 5 Name: dog

![](https://github.com/raquelredo/raquelredo.github.io_site/blob/master/_portfolio/DL-Image-classification/image1.png?raw=true)


## Preprocess Functions
The `normalize` function will take in image data, x, and return it as a normalized Numpy array. The values will be in the range of 0 to 1, inclusive. The return object will be the same shape as x.

### Normalize
```python
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # Implement Function
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min)

tests.test_normalize(normalize)
```
Tests Passed

### One-hot encode
Implement the function to return the list of labels as One-Hot encoded Numpy array. The possible values for labels are 0 to 9. The `one-hot` encoding function should return the same encoding for each value between each call to `one_hot_encode`. Make sure to save the map of encodings outside the function.

```python
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # Implement Function
    return np.eye(10)[x]

tests.test_one_hot_encode(one_hot_encode)
```
Tests Passed

### Preprocess all the data and save it
```python
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
```

### Check Point
I will create here a check point saving the preprocessed data to the disk.

```python
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
```
## Build the network
For the neural network, I'll build each layer into a function. Putting each layer in a function will allow test for simple mistakes using the unittests.

### Input
The neural network needs to read the image data, one-hot encoded labels, and dropout keep probability. I will implement the following functions:

+ Implement `neural_net_image_input` that will...
  + Return a [TF Placeholder][f7eb9246]
  + Set the shape using `image_shape` with batch size set to `None` (for a dynamic size).
  + Name the TensorFlow placeholder "x" using the TensorFlow `name` parameter in the [TF Placeholder][f7eb9246].

  [f7eb9246]: https://www.tensorflow.org/api_docs/python/tf/placeholder "TF Placeholder"

+ Implement `neural_net_label_input`
  + Return a [TF Placeholder][f7eb9246]
  + Set the shape using n_classes with batch size set to None.
  + Name the TensorFlow placeholder "y" using the TensorFlow name parameter in the [TF Placeholder][f7eb9246].

+ Implement neural_net_keep_prob_input
  + Return a [TF Placeholder][f7eb9246] for dropout keep probability.
  + Name the TensorFlow placeholder "keep_prob" using the TensorFlow name parameter in the [TF Placeholder][f7eb9246].

```python
import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, *image_shape], name= "x")

def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes], name= "y")

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name="keep_prob")

tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
```
Image Input Tests Passed.
Label Input Tests Passed.
Keep Prob Tests Passed.

## Convolution and Max Pooling Layer
Convolution layers have a lot of success with images. For this code cell, I will implement the function `conv2d_maxpool` to apply convolution then max pooling. I am asked to:

+ Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
+ Apply a convolution to `x_tensor` using weight and `conv_strides`.
+ Add bias
+ Add a nonlinear activation to the convolution.
+ Apply Max Pooling using `pool_ksize` and `pool_strides`.

```python
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # Implement Function
    shape = x_tensor.get_shape().as_list()[-1]
    # Weights
    weights = tf.Variable(tf.truncated_normal(
        [conv_ksize[0], conv_ksize[1], shape, conv_num_outputs], stddev=0.05))
    # Bias
    bias = tf.Variable(tf.zeros([conv_num_outputs]))
    # Conv. Layer
    strides = [1, *conv_strides, 1]
    output = tf.nn.conv2d(x_tensor, weights, strides, padding='SAME')
    activation = tf.nn.relu(tf.nn.bias_add(output, bias))
    # Max pooling
    ksize = [1, *pool_ksize, 1]
    strides = [1, *pool_strides, 1]
    max_pool = tf.nn.max_pool(activation, ksize, strides, padding='SAME')

    return max_pool

tests.test_con_pool(conv2d_maxpool)
```
Tests Passed

### Flatten Layer
I have to implement the flatten function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor. The output should be the shape (*Batch Size*, *Flattened Image Size*).

```python
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    #Implement Function
    shape = x_tensor.get_shape().as_list() # [None, 10, 30, 6]
    dimension = shape[1]*shape[2]*shape[3] # 18000
    return tf.reshape(x_tensor, [-1, (dimension)])

tests.test_flatten(flatten)
```
Tests Passed

### Fully-Connected Layer
I will, now to implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (Batch Size, num_outputs).

```python
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # Implement Function
    shape = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([shape[1], num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros(num_outputs))
    fc = tf.add(tf.matmul(x_tensor, weights), bias)
    return tf.nn.relu(fc)

tests.test_fully_conn(fully_conn)
```
Tests Passed

### Output Layer
Implement the output function to apply a fully connected layer to x_tensor with the shape (Batch Size, num_outputs). S

```python
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # Implement Function
    shape = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([shape[1], num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor, weights), bias)

tests.test_output(output)
```
Tests Passed

### Create Convolutional Model
Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, x, and outputs logits. I have to use the layers I created above to create this model:
+ Apply 1, 2, or 3 Convolution and Max Pool layers
+ Apply a Flatten Layer
+ Apply 1, 2, or 3 Fully Connected Layers
+ Apply an Output Layer
+ Return the output
+ Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`.

```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv = conv2d_maxpool(x,    32,  (8,8), (1,1), (2,2), (2,2))
    conv = conv2d_maxpool(conv, 64,  (4,4), (1,1), (2,2), (2,2))
    conv = conv2d_maxpool(conv, 256, (2,2), (1,1), (2,2), (2,2))

    # Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flat_x = flatten(conv)

    # Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc = fully_conn(flat_x, 512)
    fc = fully_conn(fc, 64)
    fc = tf.nn.dropout(fc, keep_prob)

    #  Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    outp = tf.nn.dropout(output(fc, 10), keep_prob)

    #return output
    return outp

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)
```
Neural Network Built!

## Train the Neural Network
### Single Optimization
I have to implement the function `train_neural_network` to do a single optimization. The optimization should use optimizer to optimize in session with a `feed_dict` of the following:
+ x for image input
+ y for labels
+ `keep_prob` for keep probability for dropout

This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.

```python
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # Implement Function
    session.run(optimizer, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: keep_probability
    })

tests.test_train_nn(train_neural_network)
```
Tests Passed

### Show Stats
I have to implement the function `print_stats` to print loss and validation accuracy. Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy. I am asked to use a keep probability of 1.0 to calculate the loss and validation accuracy.

```python
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # Implement Function
    loss = session.run(cost, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.0
    })

    train_acc = session.run(accuracy, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.0
    })

    valid_acc = session.run(accuracy, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1.0
    })

    print('Loss: {:>10.4f} , Training Acc: {:.4f}, Validation Acc: {:.4f}'
          .format( loss, train_acc, valid_acc))
```
### Hyperparameters
Tune the following parameters:
+ Set `epochs` to the number of iterations until the network stops learning or start overfitting
+ Set `batch_size` to the highest number that your machine has memory for. Most people set them to common sizes of memory:
+ 64
+ 128
+ 256
+ ...
+ Set `keep_probability` to the probability of keeping a node using dropout

```python
# Tune Parameters
epochs = 25
batch_size = 1000
keep_probability = 0.8
```

### Train on a Single CIFAR-10 Batch
Instead of training the neural network on all the CIFAR-10 batches of data, let's use a single batch. This should save time while you iterate on the model to get a better accuracy. Once the final validation accuracy is 50% or greater, run the model on all the data in the next section.

```python
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
```

Checking the Training on a Single Batch...
Epoch  1, CIFAR-10 Batch 1:  Loss:     2.1735 , Training Acc: 0.2570, Validation Acc: 0.2386
Epoch  2, CIFAR-10 Batch 1:  Loss:     1.9695 , Training Acc: 0.3040, Validation Acc: 0.3170
Epoch  3, CIFAR-10 Batch 1:  Loss:     1.9200 , Training Acc: 0.3350, Validation Acc: 0.3384
Epoch  4, CIFAR-10 Batch 1:  Loss:     1.8026 , Training Acc: 0.3760, Validation Acc: 0.3516
Epoch  5, CIFAR-10 Batch 1:  Loss:     1.7348 , Training Acc: 0.3930, Validation Acc: 0.4008
Epoch  6, CIFAR-10 Batch 1:  Loss:     1.7350 , Training Acc: 0.3840, Validation Acc: 0.3640
Epoch  7, CIFAR-10 Batch 1:  Loss:     1.6419 , Training Acc: 0.4340, Validation Acc: 0.4200
Epoch  8, CIFAR-10 Batch 1:  Loss:     1.5720 , Training Acc: 0.4520, Validation Acc: 0.4258
Epoch  9, CIFAR-10 Batch 1:  Loss:     1.5430 , Training Acc: 0.4670, Validation Acc: 0.4432
Epoch 10, CIFAR-10 Batch 1:  Loss:     1.4973 , Training Acc: 0.4790, Validation Acc: 0.4550
Epoch 11, CIFAR-10 Batch 1:  Loss:     1.4777 , Training Acc: 0.4810, Validation Acc: 0.4578
Epoch 12, CIFAR-10 Batch 1:  Loss:     1.4480 , Training Acc: 0.4870, Validation Acc: 0.4612
Epoch 13, CIFAR-10 Batch 1:  Loss:     1.4241 , Training Acc: 0.5070, Validation Acc: 0.4798
Epoch 14, CIFAR-10 Batch 1:  Loss:     1.3740 , Training Acc: 0.5190, Validation Acc: 0.4816
Epoch 15, CIFAR-10 Batch 1:  Loss:     1.3465 , Training Acc: 0.5350, Validation Acc: 0.4958
Epoch 16, CIFAR-10 Batch 1:  Loss:     1.3032 , Training Acc: 0.5540, Validation Acc: 0.5028
Epoch 17, CIFAR-10 Batch 1:  Loss:     1.3063 , Training Acc: 0.5520, Validation Acc: 0.4998
Epoch 18, CIFAR-10 Batch 1:  Loss:     1.2879 , Training Acc: 0.5470, Validation Acc: 0.4872
Epoch 19, CIFAR-10 Batch 1:  Loss:     1.2309 , Training Acc: 0.5850, Validation Acc: 0.5140
Epoch 20, CIFAR-10 Batch 1:  Loss:     1.2194 , Training Acc: 0.5910, Validation Acc: 0.5150
Epoch 21, CIFAR-10 Batch 1:  Loss:     1.1732 , Training Acc: 0.6080, Validation Acc: 0.5342
Epoch 22, CIFAR-10 Batch 1:  Loss:     1.1581 , Training Acc: 0.6120, Validation Acc: 0.5302
Epoch 23, CIFAR-10 Batch 1:  Loss:     1.1089 , Training Acc: 0.6360, Validation Acc: 0.5404
Epoch 24, CIFAR-10 Batch 1:  Loss:     1.0868 , Training Acc: 0.6420, Validation Acc: 0.5434
Epoch 25, CIFAR-10 Batch 1:  Loss:     1.0422 , Training Acc: 0.6500, Validation Acc: 0.5530

### Fully Train the Model
Now that I got a good accuracy with a single CIFAR-10 batch, I will try it with all five batches.

```python
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
```
Training...
Epoch  1, CIFAR-10 Batch 1:  Loss:     2.1613 , Training Acc: 0.1830, Validation Acc: 0.1720
Epoch  1, CIFAR-10 Batch 2:  Loss:     2.0472 , Training Acc: 0.2630, Validation Acc: 0.2592
Epoch  1, CIFAR-10 Batch 3:  Loss:     1.9339 , Training Acc: 0.3070, Validation Acc: 0.3124
Epoch  1, CIFAR-10 Batch 4:  Loss:     1.8589 , Training Acc: 0.3300, Validation Acc: 0.3248
Epoch  1, CIFAR-10 Batch 5:  Loss:     1.7997 , Training Acc: 0.3520, Validation Acc: 0.3704
Epoch  2, CIFAR-10 Batch 1:  Loss:     1.7476 , Training Acc: 0.3930, Validation Acc: 0.3954
Epoch  2, CIFAR-10 Batch 2:  Loss:     1.6547 , Training Acc: 0.4250, Validation Acc: 0.4196
Epoch  2, CIFAR-10 Batch 3:  Loss:     1.5970 , Training Acc: 0.4340, Validation Acc: 0.4328
Epoch  2, CIFAR-10 Batch 4:  Loss:     1.5790 , Training Acc: 0.4410, Validation Acc: 0.4402
Epoch  2, CIFAR-10 Batch 5:  Loss:     1.5562 , Training Acc: 0.4600, Validation Acc: 0.4644
Epoch  3, CIFAR-10 Batch 1:  Loss:     1.5143 , Training Acc: 0.4760, Validation Acc: 0.4426
Epoch  3, CIFAR-10 Batch 2:  Loss:     1.5532 , Training Acc: 0.4540, Validation Acc: 0.4520
Epoch  3, CIFAR-10 Batch 3:  Loss:     1.4297 , Training Acc: 0.4970, Validation Acc: 0.4848
Epoch  3, CIFAR-10 Batch 4:  Loss:     1.4657 , Training Acc: 0.4770, Validation Acc: 0.4744
Epoch  3, CIFAR-10 Batch 5:  Loss:     1.4836 , Training Acc: 0.4820, Validation Acc: 0.4746
Epoch  4, CIFAR-10 Batch 1:  Loss:     1.4235 , Training Acc: 0.5240, Validation Acc: 0.4758
Epoch  4, CIFAR-10 Batch 2:  Loss:     1.4351 , Training Acc: 0.5180, Validation Acc: 0.4894
Epoch  4, CIFAR-10 Batch 3:  Loss:     1.3425 , Training Acc: 0.5150, Validation Acc: 0.5044
Epoch  4, CIFAR-10 Batch 4:  Loss:     1.3640 , Training Acc: 0.5400, Validation Acc: 0.5198
Epoch  4, CIFAR-10 Batch 5:  Loss:     1.3760 , Training Acc: 0.5200, Validation Acc: 0.5048
Epoch  5, CIFAR-10 Batch 1:  Loss:     1.3118 , Training Acc: 0.5500, Validation Acc: 0.5102
Epoch  5, CIFAR-10 Batch 2:  Loss:     1.4052 , Training Acc: 0.5050, Validation Acc: 0.5008
Epoch  5, CIFAR-10 Batch 3:  Loss:     1.2889 , Training Acc: 0.5280, Validation Acc: 0.5162
Epoch  5, CIFAR-10 Batch 4:  Loss:     1.2964 , Training Acc: 0.5600, Validation Acc: 0.5176
Epoch  5, CIFAR-10 Batch 5:  Loss:     1.2850 , Training Acc: 0.5740, Validation Acc: 0.5352
Epoch  6, CIFAR-10 Batch 1:  Loss:     1.2713 , Training Acc: 0.5660, Validation Acc: 0.5230
Epoch  6, CIFAR-10 Batch 2:  Loss:     1.2860 , Training Acc: 0.5530, Validation Acc: 0.5422
Epoch  6, CIFAR-10 Batch 3:  Loss:     1.2101 , Training Acc: 0.5650, Validation Acc: 0.5540
Epoch  6, CIFAR-10 Batch 4:  Loss:     1.2300 , Training Acc: 0.5840, Validation Acc: 0.5568
Epoch  6, CIFAR-10 Batch 5:  Loss:     1.2172 , Training Acc: 0.5900, Validation Acc: 0.5588
Epoch  7, CIFAR-10 Batch 1:  Loss:     1.1583 , Training Acc: 0.6070, Validation Acc: 0.5732
Epoch  7, CIFAR-10 Batch 2:  Loss:     1.2025 , Training Acc: 0.5860, Validation Acc: 0.5678
Epoch  7, CIFAR-10 Batch 3:  Loss:     1.1282 , Training Acc: 0.5950, Validation Acc: 0.5696
Epoch  7, CIFAR-10 Batch 4:  Loss:     1.1515 , Training Acc: 0.6130, Validation Acc: 0.5854
Epoch  7, CIFAR-10 Batch 5:  Loss:     1.1513 , Training Acc: 0.6120, Validation Acc: 0.5694
Epoch  8, CIFAR-10 Batch 1:  Loss:     1.1141 , Training Acc: 0.6300, Validation Acc: 0.5872
Epoch  8, CIFAR-10 Batch 2:  Loss:     1.1385 , Training Acc: 0.6080, Validation Acc: 0.5808
Epoch  8, CIFAR-10 Batch 3:  Loss:     1.0713 , Training Acc: 0.6200, Validation Acc: 0.5948
Epoch  8, CIFAR-10 Batch 4:  Loss:     1.1033 , Training Acc: 0.6200, Validation Acc: 0.6096
Epoch  8, CIFAR-10 Batch 5:  Loss:     1.0830 , Training Acc: 0.6390, Validation Acc: 0.5930
Epoch  9, CIFAR-10 Batch 1:  Loss:     1.0889 , Training Acc: 0.6340, Validation Acc: 0.5938
Epoch  9, CIFAR-10 Batch 2:  Loss:     1.0564 , Training Acc: 0.6270, Validation Acc: 0.6034
Epoch  9, CIFAR-10 Batch 3:  Loss:     1.0041 , Training Acc: 0.6440, Validation Acc: 0.6038
Epoch  9, CIFAR-10 Batch 4:  Loss:     1.0425 , Training Acc: 0.6390, Validation Acc: 0.6122
Epoch  9, CIFAR-10 Batch 5:  Loss:     1.0123 , Training Acc: 0.6740, Validation Acc: 0.6178
Epoch 10, CIFAR-10 Batch 1:  Loss:     1.0151 , Training Acc: 0.6590, Validation Acc: 0.6126
Epoch 10, CIFAR-10 Batch 2:  Loss:     0.9958 , Training Acc: 0.6550, Validation Acc: 0.6154
Epoch 10, CIFAR-10 Batch 3:  Loss:     0.9515 , Training Acc: 0.6710, Validation Acc: 0.6204
Epoch 10, CIFAR-10 Batch 4:  Loss:     1.0085 , Training Acc: 0.6480, Validation Acc: 0.6232
Epoch 10, CIFAR-10 Batch 5:  Loss:     0.9862 , Training Acc: 0.6760, Validation Acc: 0.6238
Epoch 11, CIFAR-10 Batch 1:  Loss:     0.9699 , Training Acc: 0.6670, Validation Acc: 0.6174
Epoch 11, CIFAR-10 Batch 2:  Loss:     0.9954 , Training Acc: 0.6470, Validation Acc: 0.6090
Epoch 11, CIFAR-10 Batch 3:  Loss:     0.9470 , Training Acc: 0.6790, Validation Acc: 0.6266
Epoch 11, CIFAR-10 Batch 4:  Loss:     0.9780 , Training Acc: 0.6690, Validation Acc: 0.6306
Epoch 11, CIFAR-10 Batch 5:  Loss:     0.9701 , Training Acc: 0.6930, Validation Acc: 0.6272
Epoch 12, CIFAR-10 Batch 1:  Loss:     0.8908 , Training Acc: 0.7000, Validation Acc: 0.6450
Epoch 12, CIFAR-10 Batch 2:  Loss:     0.8981 , Training Acc: 0.6940, Validation Acc: 0.6474
Epoch 12, CIFAR-10 Batch 3:  Loss:     0.8604 , Training Acc: 0.7090, Validation Acc: 0.6564
Epoch 12, CIFAR-10 Batch 4:  Loss:     0.9404 , Training Acc: 0.6820, Validation Acc: 0.6320
Epoch 12, CIFAR-10 Batch 5:  Loss:     0.9584 , Training Acc: 0.6870, Validation Acc: 0.6190
Epoch 13, CIFAR-10 Batch 1:  Loss:     0.8609 , Training Acc: 0.7150, Validation Acc: 0.6498
Epoch 13, CIFAR-10 Batch 2:  Loss:     0.9064 , Training Acc: 0.6840, Validation Acc: 0.6450
Epoch 13, CIFAR-10 Batch 3:  Loss:     0.8624 , Training Acc: 0.7010, Validation Acc: 0.6576
Epoch 13, CIFAR-10 Batch 4:  Loss:     0.8858 , Training Acc: 0.6950, Validation Acc: 0.6482
Epoch 13, CIFAR-10 Batch 5:  Loss:     0.8765 , Training Acc: 0.7190, Validation Acc: 0.6464
Epoch 14, CIFAR-10 Batch 1:  Loss:     0.8255 , Training Acc: 0.7230, Validation Acc: 0.6552
Epoch 14, CIFAR-10 Batch 2:  Loss:     0.8174 , Training Acc: 0.7190, Validation Acc: 0.6690
Epoch 14, CIFAR-10 Batch 3:  Loss:     0.7931 , Training Acc: 0.7360, Validation Acc: 0.6724
Epoch 14, CIFAR-10 Batch 4:  Loss:     0.8097 , Training Acc: 0.7360, Validation Acc: 0.6666
Epoch 14, CIFAR-10 Batch 5:  Loss:     0.8281 , Training Acc: 0.7330, Validation Acc: 0.6496
Epoch 15, CIFAR-10 Batch 1:  Loss:     0.7788 , Training Acc: 0.7380, Validation Acc: 0.6592
Epoch 15, CIFAR-10 Batch 2:  Loss:     0.8098 , Training Acc: 0.7190, Validation Acc: 0.6666
Epoch 15, CIFAR-10 Batch 3:  Loss:     0.7741 , Training Acc: 0.7480, Validation Acc: 0.6634
Epoch 15, CIFAR-10 Batch 4:  Loss:     0.7806 , Training Acc: 0.7390, Validation Acc: 0.6686
Epoch 15, CIFAR-10 Batch 5:  Loss:     0.7589 , Training Acc: 0.7660, Validation Acc: 0.6724
Epoch 16, CIFAR-10 Batch 1:  Loss:     0.7288 , Training Acc: 0.7540, Validation Acc: 0.6728
Epoch 16, CIFAR-10 Batch 2:  Loss:     0.7584 , Training Acc: 0.7340, Validation Acc: 0.6750
Epoch 16, CIFAR-10 Batch 3:  Loss:     0.7098 , Training Acc: 0.7630, Validation Acc: 0.6770
Epoch 16, CIFAR-10 Batch 4:  Loss:     0.7426 , Training Acc: 0.7540, Validation Acc: 0.6746
Epoch 16, CIFAR-10 Batch 5:  Loss:     0.7299 , Training Acc: 0.7600, Validation Acc: 0.6748
Epoch 17, CIFAR-10 Batch 1:  Loss:     0.6846 , Training Acc: 0.7780, Validation Acc: 0.6770
Epoch 17, CIFAR-10 Batch 2:  Loss:     0.7288 , Training Acc: 0.7600, Validation Acc: 0.6840
Epoch 17, CIFAR-10 Batch 3:  Loss:     0.6836 , Training Acc: 0.7790, Validation Acc: 0.6712
Epoch 17, CIFAR-10 Batch 4:  Loss:     0.7124 , Training Acc: 0.7370, Validation Acc: 0.6716
Epoch 17, CIFAR-10 Batch 5:  Loss:     0.7087 , Training Acc: 0.7650, Validation Acc: 0.6750
Epoch 18, CIFAR-10 Batch 1:  Loss:     0.6779 , Training Acc: 0.7750, Validation Acc: 0.6764
Epoch 18, CIFAR-10 Batch 2:  Loss:     0.7126 , Training Acc: 0.7680, Validation Acc: 0.6776
Epoch 18, CIFAR-10 Batch 3:  Loss:     0.6848 , Training Acc: 0.7550, Validation Acc: 0.6606
Epoch 18, CIFAR-10 Batch 4:  Loss:     0.6878 , Training Acc: 0.7640, Validation Acc: 0.6768
Epoch 18, CIFAR-10 Batch 5:  Loss:     0.6341 , Training Acc: 0.8030, Validation Acc: 0.6914
Epoch 19, CIFAR-10 Batch 1:  Loss:     0.6539 , Training Acc: 0.7840, Validation Acc: 0.6752
Epoch 19, CIFAR-10 Batch 2:  Loss:     0.6902 , Training Acc: 0.7860, Validation Acc: 0.6704
Epoch 19, CIFAR-10 Batch 3:  Loss:     0.6293 , Training Acc: 0.7690, Validation Acc: 0.6836
Epoch 19, CIFAR-10 Batch 4:  Loss:     0.6399 , Training Acc: 0.7770, Validation Acc: 0.6828
Epoch 19, CIFAR-10 Batch 5:  Loss:     0.6364 , Training Acc: 0.8080, Validation Acc: 0.6812
Epoch 20, CIFAR-10 Batch 1:  Loss:     0.5851 , Training Acc: 0.7980, Validation Acc: 0.6786
Epoch 20, CIFAR-10 Batch 2:  Loss:     0.6570 , Training Acc: 0.7790, Validation Acc: 0.6722
Epoch 20, CIFAR-10 Batch 3:  Loss:     0.6035 , Training Acc: 0.7960, Validation Acc: 0.6766
Epoch 20, CIFAR-10 Batch 4:  Loss:     0.6086 , Training Acc: 0.8100, Validation Acc: 0.6762
Epoch 20, CIFAR-10 Batch 5:  Loss:     0.6333 , Training Acc: 0.8070, Validation Acc: 0.6766
Epoch 21, CIFAR-10 Batch 1:  Loss:     0.5789 , Training Acc: 0.8080, Validation Acc: 0.6932
Epoch 21, CIFAR-10 Batch 2:  Loss:     0.6093 , Training Acc: 0.8140, Validation Acc: 0.6828
Epoch 21, CIFAR-10 Batch 3:  Loss:     0.5716 , Training Acc: 0.8020, Validation Acc: 0.6738
Epoch 21, CIFAR-10 Batch 4:  Loss:     0.5462 , Training Acc: 0.8110, Validation Acc: 0.6886
Epoch 21, CIFAR-10 Batch 5:  Loss:     0.5733 , Training Acc: 0.8190, Validation Acc: 0.6844
Epoch 22, CIFAR-10 Batch 1:  Loss:     0.5393 , Training Acc: 0.8200, Validation Acc: 0.6956
Epoch 22, CIFAR-10 Batch 2:  Loss:     0.6131 , Training Acc: 0.8020, Validation Acc: 0.6752
Epoch 22, CIFAR-10 Batch 3:  Loss:     0.5595 , Training Acc: 0.8060, Validation Acc: 0.6754
Epoch 22, CIFAR-10 Batch 4:  Loss:     0.5474 , Training Acc: 0.8130, Validation Acc: 0.6708
Epoch 22, CIFAR-10 Batch 5:  Loss:     0.5470 , Training Acc: 0.8320, Validation Acc: 0.6886
Epoch 23, CIFAR-10 Batch 1:  Loss:     0.5020 , Training Acc: 0.8410, Validation Acc: 0.6988
Epoch 23, CIFAR-10 Batch 2:  Loss:     0.5946 , Training Acc: 0.8170, Validation Acc: 0.6660
Epoch 23, CIFAR-10 Batch 3:  Loss:     0.6193 , Training Acc: 0.7850, Validation Acc: 0.6492
Epoch 23, CIFAR-10 Batch 4:  Loss:     0.5592 , Training Acc: 0.8200, Validation Acc: 0.6730
Epoch 23, CIFAR-10 Batch 5:  Loss:     0.5190 , Training Acc: 0.8450, Validation Acc: 0.6874
Epoch 24, CIFAR-10 Batch 1:  Loss:     0.4753 , Training Acc: 0.8300, Validation Acc: 0.7012
Epoch 24, CIFAR-10 Batch 2:  Loss:     0.5532 , Training Acc: 0.8190, Validation Acc: 0.6828
Epoch 24, CIFAR-10 Batch 3:  Loss:     0.5394 , Training Acc: 0.8040, Validation Acc: 0.6668
Epoch 24, CIFAR-10 Batch 4:  Loss:     0.5323 , Training Acc: 0.8230, Validation Acc: 0.6810
Epoch 24, CIFAR-10 Batch 5:  Loss:     0.4787 , Training Acc: 0.8420, Validation Acc: 0.6898
Epoch 25, CIFAR-10 Batch 1:  Loss:     0.4927 , Training Acc: 0.8370, Validation Acc: 0.6922
Epoch 25, CIFAR-10 Batch 2:  Loss:     0.5253 , Training Acc: 0.8270, Validation Acc: 0.6950
Epoch 25, CIFAR-10 Batch 3:  Loss:     0.4446 , Training Acc: 0.8430, Validation Acc: 0.6866
Epoch 25, CIFAR-10 Batch 4:  Loss:     0.4802 , Training Acc: 0.8450, Validation Acc: 0.6836
Epoch 25, CIFAR-10 Batch 5:  Loss:     0.4633 , Training Acc: 0.8480, Validation Acc: 0.6868


## Checkpoint
The model has been saved to disk.

### Test Model
Now I will test the model against the test dataset. This will be my final accuracy.

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)

test_model()
```
INFO:tensorflow:Restoring parameters from ./image_classification
Testing Accuracy: 0.6875999808311463

![](https://github.com/raquelredo/raquelredo.github.io_site/blob/master/_portfolio/DL-Image-classification/image2.png?raw=true)
