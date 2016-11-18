import tensorflow as tf
import math
import numpy as np

def ReshapeUpLayer(X, m):
    n = int(X.get_shape()[0])
    return tf.pad(X, [[0, m-n], [0, 0]], name="reshape_up")

def DizzyLayer(X):
    n = int(X.get_shape()[0])
    n_prime = n*(n-1)/2
    thetas = tf.Variable(tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas")
    X_split = [X[k, :] for k in range(n)]
    indices = [(a, b) for b in range(n) for a in range(b)]
    for k in range(n_prime):
        (a, b) = indices[k]
        theta = thetas[k]
        c = tf.cos(theta)
        s = tf.sin(theta)
        v_1 =  c*X_split[a]+s*X_split[b]
        v_2 = -s*X_split[a]+c*X_split[b]
        X_split[a] = v_1
        X_split[b] = v_2
    return tf.pack(X_split)

def DiagLayer(X, std_dev=1, Lambda=1):
    n = int(X.get_shape()[0])
    sigmas = tf.Variable(tf.random_normal([n, 1], 1, 1), name="sigmas")
    L_sigma = tf.matmul(tf.transpose(sigmas-1), sigmas-1)
    return tf.mul(sigmas, X), L_sigma

def BiasLayer(X, std_dev=1):
    n = int(X.get_shape()[0])
    b = tf.Variable(tf.random_normal([n, 1], stddev=1), name="biases")
    return tf.add(X, b)

def DecompLayer(X, std_dev=1, Lambda=1):
    X, L_sigma = DiagLayer(DizzyLayer(X), std_dev, Lambda)
    return BiasLayer(DizzyLayer(X), std_dev), L_sigma

def FullyConnectedLayer(X, m, std_dev=1):
    n = int(X.get_shape()[0])
    W = tf.Variable(tf.random_normal([m, n], stddev=std_dev))
    return tf.matmul(W, X)

def AbsLayer(X):
    return tf.abs(X)

n = 4
m = 10
X = tf.placeholder(tf.float32, [n, None], name="input")
Y = ReshapeUpLayer(X, m)
Y, L_sigma_1 = DecompLayer(Y)
Y = AbsLayer(Y)
Y, L_sigma_2 = DecompLayer(Y)
Y = AbsLayer(Y)
Y = FullyConnectedLayer(Y, 2)
Z = tf.nn.softmax(Y)
L_sigmas = tf.concat(0, [L_sigma_1, L_sigma_2])
init = tf.initialize_all_variables()

data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
with tf.Session() as sess:
    sess.run(init)
    tf.train.SummaryWriter("./wut", sess.graph)
    print sess.run([Z, L_sigmas], feed_dict={X: data})
