import tensorflow as tf
import pandas as pd
import numpy as np

datapath = "../data/white.csv"

data_sets = pd.read_csv(datapath, delimiter = ',')
data_sets.dropna(inplace = True)

x_data = data_sets.drop(columns = ["type", "quality","y_or_n",'index'],axis = 1)
x_train = x_data.values.tolist()
x_train = np.array(x_train)

y_train = []
for i in data_sets['y_or_n']:
    if(i == True):
        y_train.append([1])
    else:
        y_train.append([0])
        
y_train = np.array(y_train)

de = len(x_train[0])

x = tf.placeholder(tf.float32, shape = [None,de])
y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([de,1]))
b = tf.Variable(tf.random_normal([1]))

h = tf.sigmoid(tf.matmul(x,W) + b)

cost = -tf.reduce_mean(y * tf.log(h) + (1-y)*tf.log(1-h))
train = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(cost)

predict_y = tf.cast(h > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y,predict_y), dtype = tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed = {x : x_train, y : y_train}
for step in range(2001):
    c, a, _ = sess.run([cost, accuracy, train], feed_dict = feed)
    if(step % 100 == 0):
        print("cost : {}, accuracy : {}".format(c, a))

my_p = sess.run(predict_y, feed_dict = {x : x_train})

print("정확도는 : {} 이다.\n".format(a))