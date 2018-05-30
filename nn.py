import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scipy.io as sio


plot_dir = 'allparams_nocut_plot2/'
column_names = {0:'lag', 1:'slag', 2:'rho0', 3:'rho1', 4:'netcc0', 5:'netcc2', 6:'penalty', 7:'netED', 8:'likelihood', 9:'duration', 10:'frequency1', 11:'frequency2', 12:'Qveto', 13:'Lveto', 14:'chirp1', 15:'chirp2', 16:'strain', 17:'hrssL', 18:'hrssH', 19:'SNR1'}
column_numbers = dict((v,k) for k,v in column_names.items())
cols = list(column_numbers) + ['label', 'index', 'false_alarms']
X_signal = sio.loadmat('data/simulation_R3_space.mat')['data']
X_signal = np.append(X_signal, np.ones((len(X_signal), 1)), 1)
X_noise = sio.loadmat('data/R3_cWB_BKG_with_qveto_gp3_dp2.mat')['data']
X_noise = np.append(X_noise, np.zeros((len(X_noise), 1)), 1)
X = np.append(X_signal, X_noise, 0)
X = X[~np.isnan(X).any(axis=1)]
X = np.concatenate([X, np.arange(len(X)).reshape(-1,1), np.array([['']]*len(X))], 1)

cols_used = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 16, 17, 18, 19]
n_params = len(cols_used)
learning_rate = 0.001
epochs = 50000
batch_size = 100
hidden_layer_size = 20
n_stats = 5

for i in range(n_stats):
    X = shuffle(X)

    x_test = X[:int(len(X)/2), cols_used].astype(float)
    y_test = X[:int(len(X)/2), -3].astype(float)
    x_train = X[int(len(X)/2):, cols_used].astype(float)
    y_train = X[int(len(X)/2):, -3].astype(float)

    x = tf.placeholder(dtype = tf.float32, shape = [None, n_params])
    y = tf.placeholder(dtype = tf.int32, shape = [None, 1])

    hl1 = tf.layers.dense(x, hidden_layer_size, kernel_initializer=tf.truncated_normal_initializer(), activation=tf.nn.sigmoid)
    y_ = tf.layers.dense(hl1, 1, kernel_initializer=tf.truncated_normal_initializer(), activation=tf.nn.sigmoid)
    loss = tf.losses.sigmoid_cross_entropy(y, y_)
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    output = tf.cast(y_ + 0.5, tf.int32)
    correct_prediction = tf.add(tf.multiply(output, y), tf.multiply(1-output, 1-y))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    saver = tf.train.Saver()
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test.reshape(-1,1)})
    print("Initial prediction accuracy on test data: ", 100*acc, "%")
    c = 1
    convergence = range(10)
    for i in range(epochs):
        _, c = sess.run([optimiser, loss], 
                     feed_dict={x: x_train, y: y_train.reshape(-1,1)})
        if (i%1000 == 0):
            acc = sess.run(accuracy,
                           feed_dict={x: x_train, y: y_train.reshape(-1,1)})
            convergence.append(acc)
            convergence = convergence[1:]
            if np.all(convergence == convergence[0]):
                break
            print("Loss = ", c, ", Prediction accuracy = ", 100*acc, "%")
    acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test.reshape(-1,1)})
    pred = sess.run(output, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
    prob = sess.run(y_, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
    cp = sess.run(correct_prediction, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
    print("Prediction accuracy on test data: ", 100*acc, "%")

    x_test_all = X[:int(len(X)/2), :]
    x_train_all = X[int(len(X)/2):, :]

    test_signal = x_test_all[np.where(y_test == 1)]
    test_noise = x_test_all[np.where(y_test == 0)]

    save_path = saver.save(sess, "checkpoints/model.ckpt")
    print("Model saved in path: %s" % save_path)

    # weights = tf.get_default_graph().get_tensor_by_name(hl1.name.split('/')[0]+'/kernel:0')
    # weights2 = tf.get_default_graph().get_tensor_by_name(y_.name.split('/')[0]+'/kernel:0')
    # w = sess.run(weights)
    # w2 = sess.run(weights2)

    sess.close()

    X[np.where((pred - y_test) == 1)[0], -1] += str(i)
    # fp = x_test_all[np.where((pred - y_test) == 1)[0], :]
    # fn = x_test_all[np.where((pred - y_test) == -1)[0], :]

    # for i in range(len(column_names)):
    #      plt.hist(test_signal[:,i], log=True, label="Signal")
    #      plt.hist(test_noise[:,i], log=True, alpha=0.7, label="Noise")
    #      plt.hist(fp[:, i], log=True, alpha=0.5, label="False positives")
    #      plt.hist(fn[:, i], log=True, alpha=0.5, label="False negatives")
    #      plt.legend()
    #      plt.xlabel(column_names[i])
    #      plt.savefig(plot_dir + column_names[i] +".png")
    #      plt.clf()
     
    # if(acc != 1):
    #     plt.hist(fp[:, column_numbers['rho1']], label="False Postives")
    #     plt.hist(fn[:, column_numbers['rho1']], alpha=0.5, label="False Negatives")
    #     plt.xlabel("$\\rho1$")
    #     plt.legend()
    #     plt.show()

    # plt.hist(prob[np.where(cp==1)], 30, log=True,label="Correct predictions")
    # plt.hist(prob[np.where(cp==0)], 30, alpha=0.5, log=True, label="Wrong predictions")
    # plt.xlabel("Confidence")
    # plt.legend()
    # plt.savefig(plot_dir + "confidence_newdata.png")
    # plt.clf()

    # plt.hist((test_signal[:, column_numbers['rho1']]), label="Signal")
    # plt.hist((test_noise[:, column_numbers['rho1']]), alpha=0.7, label="Noise")
    # plt.xlabel("$\\rho$")
    # plt.legend()
    # plt.savefig("fpfn_newdata.png")
    # plt.clf()

