from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# get mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# get next 100 images
get_next_images, get_next_labels = mnist.train.next_batch(100)
# get test images
get_test_images = mnist.test.images
# get test labels
get_test_labels = mnist.test.labels

entries = tf.placeholder(tf.float32, [None, 784])
hidden = tf.layers.dense(entries, 512, activation=tf.nn.relu)
outs = tf.layers.dense(hidden, 10)

labels = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outs, labels=labels), axis=0)
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

true = tf.placeholder(tf.float32, shape=[None, 10], name='true')
pred = tf.placeholder(tf.float32, shape=[None, 10], name='pred')
acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(true, axis=1), predictions=tf.argmax(pred, 1))

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(global_init)
    sess.run(local_init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        loss_value = sess.run(loss, feed_dict={entries: batch_xs, labels: batch_ys})
        sess.run(optimizer, feed_dict={entries: batch_xs, labels: batch_ys})
        outs2 = sess.run(tf.sigmoid(outs), feed_dict={entries: mnist.test.images})
        accuracy = sess.run(acc_op, feed_dict={true: mnist.test.labels, pred: outs2})
        if (accuracy > 0.92 and learning_rate > 0.002):
            print('Inc learning rate at', i)
            learning_rate = 0.001
        if (i % 10 == 0):
            print("Epoch : %d; Accuracy : %.2f" %(i, accuracy * 100.0))
    outs2 = sess.run(tf.sigmoid(outs), feed_dict={entries: mnist.test.images})
    accuracy = sess.run(acc_op, feed_dict={true: mnist.test.labels, pred: outs2})
    print("Accuracy : %.2f" %(accuracy * 100.0))
