
# coding: utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time

FTRAIN = './data/training.csv'

def load(test=False, cols=None):

    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    #print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 96, 96, 1)
    return X, y

X, y = load2d();

def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis, x_shape[axis], x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis, x_rep)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

class Recomb():
    def conv_filter(self, name, kw, kh, n_in, n_out):
        """
        kw, kh - filter width and height
        n_in - number of input channels
        n_out - number of output channels
        """
        kernel_init_val = tf.truncated_normal([kh, kw, n_in, n_out], dtype=tf.float32, stddev=0.1)
        kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
        return kernel

    def conv_bias(self, name, n_out):
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        return biases

    def conv_layer(self, bottom, name,kw, kh, n_out, dw=1, dh=1):

        n_in = bottom.get_shape()[-1].value

        with tf.variable_scope(name) as scope:
            filt = self.conv_filter(name, kw, kh, n_in, n_out)
            conv = tf.nn.conv2d(bottom, filt, (1, dh, dw, 1), padding='SAME')

            conv_biases = self.conv_bias(name, n_out)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.tanh(bias)
            return relu

    def mpool_op(self, bottom, name, kh=2, kw=2, dh=2, dw=2):
        return tf.nn.max_pool(bottom,
                              ksize=[1, kh, kw, 1],
                              strides=[1, dh, dw, 1],
                              padding='VALID',
                              name=name)

    def upsample(self,X,scale):
        output = repeat_elements(repeat_elements(X, scale[0], axis=1),scale[1], axis=2)
        return output


    def fc_op(self, input_op, name, n_out):
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=0.1), name='w')
            biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), name='b')
            activation = tf.nn.tanh(tf.matmul(input_op, kernel) + biases)
            return activation

    def build(self, input_im, dropout_keep_prob=0.75):
        net = {}

        net["conv1_4"] = self.conv_layer(input_im, name="conv1_4", kh=3, kw=3, n_out=16) #b x 96 x 96 x 16
        net["conv1_4_p"] = self.mpool_op(net["conv1_4"],"conv1_4") # b x 48 x 48 x 16

        net["conv1_3"] = self.conv_layer(net["conv1_4_p"], name="conv1_3", kh=3, kw=3, n_out=32)
        net["conv1_3_p"] = self.mpool_op(net["conv1_3"], "conv1_3") # b x 24 x 24 x 32

        net["conv1_2"] = self.conv_layer(net["conv1_3_p"], name="conv1_2", kh=3, kw=3, n_out=48)
        net["conv1_2_p"] = self.mpool_op(net["conv1_2"],"conv1_2")  # b x 12 x 12 x 48

        net["conv1_1"] = self.conv_layer(net["conv1_2_p"], name="conv1_1", kh=3, kw=3, n_out=48) # b x 12 x 12 x 48

        net["conv2_1"] = self.upsample(net["conv1_1"], [2,2])
        concat = tf.concat(3, [net["conv2_1"], net["conv1_2"]]) # b x 24 x 24 x 96
        net["conv2_1_c"] = self.conv_layer(
                            self.conv_layer(concat,name="conv2_1_c", kh=3, kw=3, n_out=48),\
                            name="conv2_1",kh=3,kw=3,n_out=32)  # b x 24 x 24 x 32
        net["conv2_2"] = self.upsample(net["conv2_1_c"], [2,2])
        concat = tf.concat(3, [net["conv2_2"], net["conv1_3"]]) # b x 48 x 48 x 64
        net["conv2_2_c"] = self.conv_layer(
                            self.conv_layer(concat,name="conv2_2_c", kh=3, kw=3, n_out=32),\
                            name="conv2_1",kh=3,kw=3,n_out=16) # b x 48 x 48 x 16
        net["conv2_3"] = self.upsample(net["conv2_2_c"], [2,2])
        concat = tf.concat(3, [net["conv2_3"], net["conv1_4"]]) # b x 96 x 96 x 32
        net["conv2_3_c"] = self.conv_layer(
                            self.conv_layer(concat,name="conv2_3_c", kh=3, kw=3, n_out=16),\
                            name="conv2_1",kh=3,kw=3,n_out=5)

        shp = net["conv2_3_c"].get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        flat = tf.reshape(net["conv2_3_c"], [-1, flattened_shape], name="flat")
        flat = tf.nn.dropout(flat,dropout_keep_prob, name='flat_drop')
        net["fc1"] = self.fc_op(flat, name="fc1", n_out=500)
        net["fc1_drop"] = tf.nn.dropout(net["fc1"], 0.9, name="fc1_drop")
        net["fc2"] = self.fc_op(net["fc1_drop"], name="fc2", n_out=30)
        return net


channels = 1
width, height = [96,96]
batch_size = 100
num_epochs = 10000
lr=0.01
tf.reset_default_graph()
in_images = tf.placeholder("float", [batch_size, width, height, channels])
position = tf.placeholder("float", [batch_size,y.shape[1]])
model = Recomb()
net = model.build(in_images)
last_layer = net["fc2"]
loss = tf.reduce_mean(tf.square(last_layer - position))
optimizer = tf.train.AdagradOptimizer(lr)
global_step = tf.Variable(0, name="global_step", trainable=False)
train_step = optimizer.minimize(loss, global_step=global_step)
initializer = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())
with tf.Session() as sess:
    sess.run(initializer)
    #saver.restore(sess, "model_l30.ckpt")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
            inputs, targets = batch
            result = sess.run(
                [train_step, loss],
                feed_dict = {
                    in_images: inputs,
                    position: targets,
                }
            )
            #print result
            train_err += result[1]
            train_batches += 1

            if np.isnan(result[1]):
                print("gradient vanished/exploded")
                break
            print '=',
        # Then we print the results for this epoch:
        print("\nEpoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if epoch%10 == 0:
            checkpoint_path = saver.save(sess, "model_l30.ckpt")
