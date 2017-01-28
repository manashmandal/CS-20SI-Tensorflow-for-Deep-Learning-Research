#Task 3
import pandas as pd
import tensorflow as tf
import math
import random
import numpy as np

df = pd.read_csv('data/heart.txt', sep='\t')

# Converting famhist to numerical value
df['famhist'].replace(to_replace={'Present' : 1.0, 'Absent' : 0.0}, inplace=True)

X_ = df[['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']].values
Y_ = df[['chd']].values


# Splits the dataset into train test portion
def train_test_split(x, y, splt_ratio=.4):
    split_ratio = round(x.shape[0] * splt_ratio)

    batch_size = split_ratio

    indices = np.arange(0, x.shape[0])

    np.random.shuffle(indices)

    X_train = x[indices[: split_ratio]]
    y_train = y[indices[: split_ratio]]

    X_test = x[indices[split_ratio:]]
    y_test = y[indices[split_ratio:]]

    return ((X_train, y_train), (X_test, y_test))


# Normalizes features
def feature_normalize(features):
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    return (features - mu) / sigma

# Encodes to onehot
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Getting the dataset
Train, Test = train_test_split(X_, Y_)
X_train, y_train = Train
X_test, y_test = Test

y_train = y_train.ravel()
y_test = y_test.ravel()

train_batch_size = len(y_train)
test_batch_size = len(y_test)

# Testing if split was correct
assert(len(y_train) + len(y_test) == df.shape[0])

X_train_normalized = feature_normalize(X_train)
X_test_normalized = feature_normalize(X_test)
y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

# Building the model
n_dim = X_train_normalized.shape[1]

learning_rate = 0.01
training_epochs = 500

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.ones([n_dim, 2]))
b = tf.Variable(tf.zeros([1, 2]))

init = tf.global_variables_initializer()

y_ = tf.nn.sigmoid(tf.matmul(X, W))
y_ = tf.nn.sigmoid(tf.matmul(X, W) + b)
cost_function = tf.reduce_mean(tf.reduce_sum((-Y * tf.log(y_)) - ((1 - Y) * tf.log(1 - y_)), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

cost_history = np.empty(shape=[1], dtype=float)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: X_train_normalized, Y: y_train_encoded})
        cost_history = np.append(cost_history,
                                 sess.run(cost_function, feed_dict={X: X_train_normalized, Y: y_train_encoded}))

    y_pred = sess.run(y_, feed_dict={X: X_test_normalized})
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", (sess.run(accuracy, feed_dict={X: X_test_normalized, Y: y_test_encoded})))
