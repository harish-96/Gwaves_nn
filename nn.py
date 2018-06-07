import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scipy.io as sio


plot_dir = 'tmp/'
column_names = {0:'lag', 1:'slag', 2:'rho0', 3:'rho1', 4:'netcc0', 5:'netcc2', 6:'penalty', 7:'netED', 8:'likelihood', 9:'duration', 10:'frequency1', 11:'frequency2', 12:'Qveto', 13:'Lveto', 14:'chirp1', 15:'chirp2', 16:'strain', 17:'hrssL', 18:'hrssH', 19:'SNR1'}
column_numbers = dict((v,k) for k,v in column_names.items())
cols = list(column_numbers) + ['label', 'index', 'false_alarms']
X_signal = sio.loadmat('data/simulation_R3_space.mat')['data']
X_signal = X_signal[np.where(X_signal[:, 3] > 10)]
X_signal = np.append(X_signal, np.ones((len(X_signal), 1)), 1)
X_signal = X_signal[~np.isnan(X_signal).any(axis=1)]
X_noise = sio.loadmat('data/R3_cWB_BKG_with_qveto_gp3_dp2.mat')['data']
X_noise = np.append(X_noise, np.zeros((len(X_noise), 1)), 1)
X_noise = X_noise[~np.isnan(X_noise).any(axis=1)]
X = np.append(X_signal, X_noise, 0)
X = np.concatenate([X, np.array([['']]*len(X))], 1)

cols_used = [2, 3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19]
n_params = len(cols_used)
learning_rate = 0.001
epochs = 50000
hidden_layer_size = 30
n_stats = 2

for it in range(n_stats):
    X = shuffle(X)

    x_test = X[:int(len(X)/2), cols_used].astype(float)
    y_test = X[:int(len(X)/2), -2].astype(float)
    x_train = X[int(len(X)/2):, cols_used].astype(float)
    y_train = X[int(len(X)/2):, -2].astype(float)

    x = tf.placeholder(dtype = tf.float32, shape = [None, n_params], name='input')
    y = tf.placeholder(dtype = tf.int32, shape = [None, 1], name='label')

    hl1 = tf.layers.dense(x, hidden_layer_size, kernel_initializer=tf.truncated_normal_initializer(), activation=tf.nn.sigmoid)
    y_ = tf.layers.dense(hl1, 1, kernel_initializer=tf.truncated_normal_initializer(), activation=tf.nn.sigmoid, name='prob')
    # loss = tf.losses.sigmoid_cross_entropy(y, y_)
    y_temp = tf.cast(y, tf.float32)
    loss = -tf.reduce_sum(y_temp * tf.log(y_ + 1e-20) + 1000*(1-y_temp) * tf.log(1-y_ + 1e-20))
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    output = tf.cast(y_ + 0.5, tf.int32, name='pred')
    correct_prediction = tf.add(tf.multiply(output, y), tf.multiply(1-output, 1-y), name='cp')
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    saver = tf.train.Saver()
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test.reshape(-1,1)})
    print("Training ", str(it))
    c = 1
    convergence = list(range(7))
    for i in range(epochs):
        _, c = sess.run([optimiser, loss], 
                     feed_dict={x: x_train, y: y_train.reshape(-1,1)})
        if(np.isnan(c)):
            print("Nan encountered")
            break
        if (i%1000 == 0):
            acc = sess.run(accuracy,
                           feed_dict={x: x_train, y: y_train.reshape(-1,1)})
            convergence.append(c)
            convergence = convergence[1:]
            if i%5000 == 0:
                print("\tLoss = ", c, ", Prediction accuracy = ", 100*acc, "%")
            if np.all(convergence == convergence[0]):
                break

    c, acc = sess.run([loss, accuracy], feed_dict={x:x_test, y:y_test.reshape(-1,1)})
    if(np.isnan(c)):
        print("Nan encountered")
        continue
    pred = sess.run(output, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
    prob = sess.run(y_, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
    cp = sess.run(correct_prediction, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)

    save_path = saver.save(sess, "tmp/model" + str(it) + ".ckpt")
    print("Model saved in path: %s" % save_path)

    sess.close()
    indices = np.where((pred - y_test) == 1)[0]
    print("\tTest Accuracy: ", 100*acc, "%", "FAR: ", 100*len(indices)/len(np.where(y_test==0)[0]), "%")
    for i in indices:
        if X[i, -1] == '':
            X[i, -1] = str(it)
        else:
            X[i, -1] = ','.join([X[i, -1], str(it)])

sio.savemat(plot_dir + 'stats_new.mat', {'data':X})
# idx = np.where(X[:, -1] != '')[0]
# if len(idx) > 0:
#     tmp = np.concatenate([np.tile(X[i,:], [len(X[i, -1].split(',')), 1]) for i in idx])

#     for i in range(len(column_names)):
#          plt.hist(X_signal[:,i], log=True, label="Signal")
#          plt.hist(X_noise[:,i], log=True, alpha=0.7, label="Noise")
#          plt.hist(tmp[:, i].astype(float), log=True, alpha=0.7, label="False positives")
#          plt.legend()
#          plt.xlabel(column_names[i])
#          plt.savefig(plot_dir + column_names[i] +".png")
#          plt.clf()

#     scatter = ['netcc0', 'netcc2', 'Qveto', 'Lveto', 'rho0', 'lag', 'chirp1']
#     for i in scatter:
#         for j in scatter:
#             if i != j:                                               
#                 plt.scatter(X[idx, column_numbers[i]].astype(float), 
#                             X[idx, column_numbers[j]].astype(float),
#                              c=[len(i.split(',')) for i in X[idx, -1]])
#                 plt.xlabel(i)
#                 plt.ylabel(j)
#                 plt.colorbar()
#                 plt.savefig(plot_dir + i + 'vs' + j +".png")
#                 plt.clf()
                 
     
#     # plt.hist(prob[np.where(cp==1)], 30, log=True,label="Correct predictions")
#     # plt.hist(prob[np.where(cp==0)], 30, alpha=0.5, log=True, label="Wrong predictions")
#     # plt.xlabel("Confidence")
#     # plt.legend()
#     # plt.savefig(plot_dir + "confidence_newdata.png")
#     # plt.clf()

#     # plt.hist((test_signal[:, column_numbers['rho1']]), label="Signal")
#     # plt.hist((test_noise[:, column_numbers['rho1']]), alpha=0.7, label="Noise")
#     # plt.xlabel("$\\rho$")
#     # plt.legend()
#     # plt.savefig("fpfn_newdata.png")
#     # plt.clf()

