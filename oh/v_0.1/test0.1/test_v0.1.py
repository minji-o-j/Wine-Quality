import tensorflow as tf
import pandas as pd
import numpy as np

datapath = "./data/winequalityN.csv"

data_sets = pd.read_csv(datapath, delimiter = ',')
data_sets.dropna(inplace = True)

x_data = data_sets.drop(columns = ["type", "quality"],axis = 1)
y_data = data_sets["quality"]

x_data = x_data.values.tolist()
y_data = y_data.values.tolist()

x_train = np.array(x_data, dtype = np.float32)
y_train = np.array(y_data, dtype = np.int32)
y_train = y_train.reshape([6463,1])

x_de = len(x_train[0])
classes = 10
x = tf.placeholder(tf.float32, shape = [None, 11])
y = tf.placeholder(tf.int32, shape = [None, 1])
y_one_hot = tf.one_hot(y, classes)
y_one_hot = tf.reshape(y_one_hot, [-1,classes])

W = tf.Variable(tf.random_normal([x_de, classes]))
b = tf.Variable(tf.random_normal([classes]))

logits = tf.matmul(x, W) + b
h = tf.nn.softmax(logits)

cost = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_one_hot)
train = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(cost)

predict = tf.argmax(h, 1)
acc = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(y_one_hot, 1)), dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        c, ac, _ = sess.run([cost, acc, train], feed_dict={x: x_train, y: y_train})
        if (step % 100 == 0):
            print("Step : {}, cost : {}, accuracy : {}".format(step, c, ac))