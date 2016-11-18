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

def DiagLayer(X, std_dev=1, Lambda=0):
    n = int(X.get_shape()[0])
    sigmas = tf.Variable(tf.random_normal([n, 1], 1, 1), name="sigmas")
    L_sigma = Lambda/2*tf.matmul(tf.transpose(sigmas-1), sigmas-1)
    return tf.mul(sigmas, X), L_sigma

def BiasLayer(X, std_dev=1):
    n = int(X.get_shape()[0])
    b = tf.Variable(tf.random_normal([n, 1], stddev=1), name="biases")
    return tf.add(X, b)

def DecompLayer(X, std_dev=1, Lambda=0):
    X, L_sigma = DiagLayer(DizzyLayer(X), std_dev, Lambda)
    return BiasLayer(DizzyLayer(X), std_dev), L_sigma

def FullyConnectedLayer(X, m, std_dev=1):
    n = int(X.get_shape()[0])
    W = tf.Variable(tf.random_normal([m, n], stddev=std_dev))
    return tf.matmul(W, X)

def AbsLayer(X):
    return tf.abs(X)

X = tf.placeholder(tf.float32, [2, None])
Y = tf.placeholder(tf.float32, [2, None])

# W_1 = tf.Variable(tf.random_normal([50, 2]))
# b_1 = tf.Variable(tf.random_normal([50, 1]))
# W_2 = tf.Variable(tf.random_normal([2, 50]))
# b_2 = tf.Variable(tf.random_normal([2, 1]))
# Y_hat = tf.matmul(W_2, tf.nn.relu(tf.matmul(W_1, X) + b_1)) + b_2

L_sigmas = []
Lambda = 1e-2
X_1 = ReshapeUpLayer(X, 5)
X_2, L_sigma = DecompLayer(X_1, Lambda=Lambda); X_3 = AbsLayer(X_2); L_sigmas.append(L_sigma)
X_4, L_sigma = DecompLayer(X_3, Lambda=Lambda); X_5 = AbsLayer(X_4); L_sigmas.append(L_sigma)
X_6, L_sigma = DecompLayer(X_5, Lambda=Lambda); X_7 = AbsLayer(X_6); L_sigmas.append(L_sigma)
X_8, L_sigma = DecompLayer(X_7, Lambda=Lambda); X_9 = AbsLayer(X_8); L_sigmas.append(L_sigma)
Y_hat = FullyConnectedLayer(X_9, 2)

L = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.transpose(Y_hat), tf.transpose(Y))) + tf.reduce_sum(tf.concat(0, L_sigmas))
optimizer = tf.train.GradientDescentOptimizer(1e-2)
train = optimizer.minimize(L)
correct_prediction = tf.equal(tf.argmax(Y_hat, 0), tf.argmax(Y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.initialize_all_variables()


def XOR_data(N):
    X = np.random.rand(2, N)
    Y = np.vectorize(round)(X)
    Y = abs(np.diag(np.dot(np.transpose(Y),Y))-1)
    Y = np.array([Y, 1-Y])
    return X, Y

X_train, Y_train = XOR_data(1000)
X_valid, Y_valid = XOR_data(100)

with tf.Session() as sess:
    # tf.train.SummaryWriter("./wut", sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={X: X_train, Y: Y_train})
        print sess.run(accuracy, feed_dict={X: X_valid, Y: Y_valid})
    # print sess.run(Y_hat, feed_dict={X: X_train, Y: Y_train})
    # print sess.run(L, feed_dict={X: X_train, Y: Y_train})
