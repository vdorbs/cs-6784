import tensorflow as tf
import math
import numpy as np

def ReshapeUpLayer(X, n, m):
    return tf.pad(X, [[0, m-n], [0, 0]], name="reshape_up")

def gen_rot_list(n):
    arr = [[0] * n for i in range(n-1)]
    rot_list = [[] for i in range(n-1)]
    idx = 0
    for i in range(n-1):
        for j in range(i+1, n):
            while arr[idx][i] == 1:
                idx = (idx+1) % (n-1)
            arr[idx][i] = 1
            arr[idx][j] = 1
            rot_list[idx].append((i, j))
    return rot_list

def DizzyLayer(X, n):
    n_prime = n*(n-1)/2
    thetas = tf.Variable(tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas")

    rot_list = gen_rot_list(n)
    results = [X]
    k = 0
    for sublist in gen_rot_list(n):
        indices = []
        values = []
        for (a, b) in sublist:
            c = tf.cos(thetas[k])
            s = tf.sin(thetas[k])
            indices = indices + [[a, a], [a, b], [b, a], [b, b]]
            values = values + [c, s, -s, c]
            k += 1
        shape = [n, n]
        v = tf.pack(tf.squeeze(values))
        R = tf.SparseTensor(indices, v, shape)
        results.append(tf.sparse_tensor_dense_matmul(R, results[-1]))
    return results[-1]


    # X_split = [X[k, :] for k in range(n)]
    # indices = [(a, b) for b in range(n) for a in range(b)]
    # for k in range(n_prime):
    #     (a, b) = indices[k]
    #     theta = thetas[k]
    #     c = tf.cos(theta)
    #     s = tf.sin(theta)
    #     v_1 =  c*X_split[a]+s*X_split[b]
    #     v_2 = -s*X_split[a]+c*X_split[b]
    #     X_split[a] = v_1
    #     X_split[b] = v_2
    # return tf.pack(X_split)


def DiagLayer(X, n, std_dev=1, Lambda=0):
    sigmas = tf.Variable(tf.random_normal([n, 1], 1, 1), name="sigmas")
    L_sigma = Lambda/2*tf.matmul(tf.transpose(sigmas-tf.ones_like(sigmas)), sigmas-tf.ones_like(sigmas))
    return tf.mul(sigmas, X), L_sigma, sigmas

def BiasLayer(X, n, std_dev=1):
    b = tf.Variable(tf.random_normal([n, 1], stddev=1), name="biases")
    return tf.add(X, b)

def DecompLayer(X, n, std_dev=1, Lambda=0):
    X, L_sigma, sigmas = DiagLayer(DizzyLayer(X, n), n, std_dev, Lambda)
    return BiasLayer(DizzyLayer(X, n), std_dev), L_sigma, sigmas

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
Lambda = 1e1
N = 6
X_1 = ReshapeUpLayer(X, 2, N)
X_2, L_sigma, sigmas_1 = DecompLayer(X_1, N, Lambda=Lambda); X_3 = AbsLayer(X_2); L_sigmas.append(L_sigma)
X_4, L_sigma, sigmas_2 = DecompLayer(X_3, N, Lambda=Lambda); X_5 = AbsLayer(X_4); L_sigmas.append(L_sigma)
X_6, L_sigma, sigmas_3 = DecompLayer(X_5, N, Lambda=Lambda); X_7 = AbsLayer(X_6); L_sigmas.append(L_sigma)
X_8, L_sigma, sigmas_4 = DecompLayer(X_7, N, Lambda=Lambda); X_9 = AbsLayer(X_8); L_sigmas.append(L_sigma)
W = tf.Variable(tf.random_normal([2, N]))
sigmas_fc = tf.svd(W, compute_uv=False)
Y_hat = tf.matmul(W, X_9)

L = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.transpose(Y_hat), tf.transpose(Y))) + tf.reduce_sum(tf.concat(0, L_sigmas))
optimizer = tf.train.GradientDescentOptimizer(1e-2)
train = optimizer.minimize(L)
correct_prediction = tf.equal(tf.argmax(Y_hat, 0), tf.argmax(Y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

training_accuracy = tf.scalar_summary("accuracy_train", accuracy)
sigmas_1_hist = tf.histogram_summary("sigmas_1", sigmas_1)
sigmas_2_hist = tf.histogram_summary("sigmas_2", sigmas_2)
sigmas_3_hist = tf.histogram_summary("sigmas_3", sigmas_3)
sigmas_4_hist = tf.histogram_summary("sigmas_4", sigmas_4)
sigmas_fc_hist = tf.histogram_summary("sigmas_fc", sigmas_fc)
training_summaries = tf.merge_summary([training_accuracy, sigmas_1_hist, sigmas_2_hist, sigmas_3_hist, sigmas_4_hist, sigmas_fc_hist])

test_accuracy = tf.scalar_summary("accuracy_test", accuracy)

init = tf.initialize_all_variables()

def XOR_data(N):
    X = np.random.rand(2, N)
    Y = np.vectorize(round)(X)
    Y = abs(np.diag(np.dot(np.transpose(Y),Y))-1)
    Y = np.array([Y, 1-Y])
    return X, Y

X_train, Y_train = XOR_data(1000)
X_test, Y_test = XOR_data(1000)

with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter("./wut", sess.graph)
    sess.run(init)
    for i in range(2500):
        print i
        training_summary, _ = sess.run([training_summaries, train], feed_dict={X: X_train, Y: Y_train})
        test_summary, _ = sess.run([test_accuracy, accuracy], feed_dict={X: X_test, Y: Y_test})
        summary_writer.add_summary(training_summary, i)
        summary_writer.add_summary(test_summary, i)
