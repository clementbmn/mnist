from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# get mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# get next 100 images
get_next_images, get_next_labels = mnist.train.next_batch(100)

# get test images
get_test_images = mnist.test.images
# get test labels
get_test_labels = mnist.test.labels

entries = tf.placeholder(tf.float32, [None, 784])
hidden = tf.layers.dense(entries, 784, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden, 512, activation=tf.nn.relu)
outs = tf.layers.dense(hidden2, 10)

labels = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outs, labels=labels), axis=0)
optimizer = tf.train.AdamOptimizer(0.0002)
op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        get_next_images, get_next_labels = mnist.train.next_batch(5000)
        sess.run(loss, feed_dict={entries: get_next_images, labels: get_next_labels})
        sess.run(op, feed_dict={entries: get_next_images, labels: get_next_labels})
        outs = sess.run(tf.sigmoid(outs), feed_dict={entries: get_test_images})
        ok = 0.
        total = 0.
        for i in range(len(outs)):
            maxOut = np.argmax(outs[i])
            maxTest = np.argmax(get_test_labels[i])
            if maxOut == maxTest:
                ok += 1.
            total += 1.
        print("Accuracy : %.2f%%" % ((ok * 1000 / total) / 10.0))
ok = 0.
total = 0.
for i in range(len(outs)):
    maxOut = np.argmax(outs[i])
    maxTest = np.argmax(get_test_labels[i])
    if maxOut == maxTest:
        ok += 1.
    total += 1.
print("Accuracy : %.2f%%" % ((ok * 1000 / total) / 10.0))
