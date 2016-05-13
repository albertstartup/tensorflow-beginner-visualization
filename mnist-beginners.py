import tensorflow as tf
import colorsys
import numpy as np
import png
import math
from functools import partial
#import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

## custom start

def sigmoidNormalize(val, mean, std):
    x = (-(val-mean))/std
    return (1)/(1 + math.exp(x))

def hsv(color, brightness):
    if color == "blue":
        return 0.6666666666666666, 1.0, brightness
    elif color == "red":
        return 0.0, 1.0, brightness

def weightToRgb(weight):
    global posNormalizer
    global negNormalizer
    global hsv
    if weight == 0:
        return [0,0,0]
    if weight > 0:
        brightness = posNormalizer(weight)
        hsvTuple = hsv(color="blue", brightness=brightness)
        return [x*255 for x in colorsys.hsv_to_rgb(*hsvTuple)]
    if weight < 0:
        brightness = negNormalizer(weight)
        hsvTuple = hsv(color="red", brightness=brightness)
        return [x*255 for x in colorsys.hsv_to_rgb(*hsvTuple)]

negNormalizer = 0
posNormalizer = 0
def weights_to_png():
    global negNormalizer
    global posNormalizer
    finW = sess.run(W)
    flatW = finW.ravel(order="F") # (10, 784) -> (7840)
    posW = [x for x in flatW if x > 0]
    negW = [x for x in flatW if x < 0]
    posMean = np.mean(posW)
    negMean = np.mean(negW)
    posStd = np.std(posW)
    negStd = np.std(negW)
    negNormalizer = partial(sigmoidNormalize, mean=negMean, std=negStd)
    posNormalizer = partial(sigmoidNormalize, mean=posMean, std=posStd)
    rgbW = map(weightToRgb, flatW) # (7840) -> (7840, 3)
    flatRgbW = np.ravel(rgbW, order="C") # (7840, 3) -> 23520
    img = flatRgbW.reshape(10, 28, 84, order="C") # 23520 -> (10, 28, 84)
    for i in range(np.shape(img)[0]):
        f = open("weights-for-%i.png" % i, "wb")
        w = png.Writer(28, 28)
        w.write(f, img[i])
        f.close()

## custom end

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

weights_to_png()
