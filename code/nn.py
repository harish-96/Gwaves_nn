
#Importing required modules
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scipy.io as sio
import os

# Directories to store plots and checkpoints
plot_dir = 'results/train_no_cuts/'
ckpts = plot_dir + "checkpoints/"
if not os.path.exists(ckpts):
    os.makedirs(ckpts)

# Input data path on disk
signal_path = 'data/simulation_R3_space.mat'
noise_path = 'data/R3_cWB_BBH_BKG_with_rho_g6cor.mat'
# noise_path = 'data/R3_cWB_BKG_with_qveto_gp3_dp2.mat'

# The following columns are present in the input data. Edit only if the input structure is changed
column_names = {0:'lag', 1:'slag', 2:'rho0', 3:'rho1', 4:'netcc0', 5:'netcc2', 6:'penalty', 7:'netED', 8:'likelihood', 9:'duration', 10:'frequency1', 11:'frequency2', 12:'Qveto', 13:'Lveto', 14:'chirp1', 15:'chirp2', 16:'strain', 17:'hrssL', 18:'hrssH', 19:'SNR1'}
column_numbers = dict((v,k) for k,v in column_names.items())

cols = list(column_numbers) + ['label', 'false_alarms']

# Loading signal and noise data
X_signal = sio.loadmat(signal_path)['data']
X_noise = sio.loadmat(noise_path)['data']

# Chosing only those events with rho1 > 10 and removing events with NaNs
X_signal = X_signal[np.where(X_signal[:, column_numbers['rho1']] > 10)]
X_signal = X_signal[~np.isnan(X_signal).any(axis=1)]
X_noise = X_noise[~np.isnan(X_noise).any(axis=1)]

# Adding labels (1 for signal and 0 for noise)
X_signal = np.append(X_signal, np.ones((len(X_signal), 1)), 1)
X_noise = np.append(X_noise, np.zeros((len(X_noise), 1)), 1)

# Concatenating the labeled data. The last column of this 2D array keeps track of the number of times an event is misclassified
X = np.append(X_signal, X_noise, 0)
X = np.concatenate([X, np.array([['']]*len(X))], 1)

# Parameters which input to the network
cols_used = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
n_params = len(cols_used)

# Training hyperparameters, network size and number of training runs over which statistics are collected
learning_rate = 0.003
epochs = 30000
hidden_layer_size = 30
n_stats = 50


# Building the tensorflow computational graph
# x and y : input data and labels
# hl1: Hidden layer's output
# y_: Network output in [0, 1]
# loss: Cross entropy cost
# output: Binary network output obtained after thresholding y_ at 0.5
# correct_prediction: equals 1 for correctly classified event and 0 for misclassified event
with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
    x = tf.placeholder(dtype = tf.float32, shape = [None, n_params], name='input')
    y = tf.placeholder(dtype = tf.int32, shape = [None, 1], name='label')

    hl1 = tf.layers.dense(x, hidden_layer_size, kernel_initializer=tf.truncated_normal_initializer(), activation=tf.nn.sigmoid)
    y_ = tf.layers.dense(hl1, 1, kernel_initializer=tf.truncated_normal_initializer(), activation=tf.nn.sigmoid, name='prob')
    y_temp = tf.cast(y, tf.float32)
    loss = -tf.reduce_sum(y_temp * tf.log(y_ + 1e-20) + 1000*(1-y_temp) * tf.log(1-y_ + 1e-20))
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    output = tf.cast(y_ + 0.5, tf.int32, name='pred')
    correct_prediction = tf.add(tf.multiply(output, y), tf.multiply(1-output, 1-y), name='cp')
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))


# Performing multiple training runs on the computational graph built above. This is to obtain statistics
for it in range(n_stats):
    X = shuffle(X)

    # Test and train data separation (50-50)
    x_test = X[:int(len(X)/2), cols_used].astype(float)
    y_test = X[:int(len(X)/2), -2].astype(float)
    x_train = X[int(len(X)/2):, cols_used].astype(float)
    y_train = X[int(len(X)/2):, -2].astype(float)

    # Begin TensorFlow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test.reshape(-1,1)})
    print("Training ", str(it))

    # Convergence check. If the loss does not change in 7000 iterations, stop training.
    convergence = list(range(7))
    for i in range(epochs):
        _, c = sess.run([optimiser, loss], 
                     feed_dict={x: x_train, y: y_train.reshape(-1,1)})
        if (i%1000 == 0):
            acc = sess.run(accuracy,
                           feed_dict={x: x_train, y: y_train.reshape(-1,1)})
            #FIFO queue
            convergence.append(c)
            convergence = convergence[1:]

            #Print current state
            print("\tLoss = ", c, ", Prediction accuracy = ", 100*acc, "%")
            if np.all(convergence == convergence[0]):
                break

    # Feed test data to trained network
    c, acc = sess.run([loss, accuracy], feed_dict={x:x_test, y:y_test.reshape(-1,1)})
    pred = sess.run(output, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
    prob = sess.run(y_, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
    cp = sess.run(correct_prediction, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)

    # Save trained model
    saver = tf.train.Saver()
    save_path = saver.save(sess, ckpts + "/model" + str(it) + ".ckpt")
    print("Model saved in path: %s" % save_path)

    # End TensorFlow session
    sess.close()

    # Indices of false alarm events
    indices = np.where((pred - y_test) == 1)[0]

    print("\tTest Accuracy: ", 100*acc, "%", "FAR: ", 100*len(indices)/len(np.where(y_test==0)[0]), "%")
    for i in indices:
        if X[i, -1] == '':
            X[i, -1] = str(it)
        else:
            X[i, -1] = ','.join([X[i, -1], str(it)])

sio.savemat(plot_dir + 'stats_new.mat', {'data':X})
idx = np.where(X[:, -1] != '')[0]
if len(idx) > 0:
    tmp = np.concatenate([np.tile(X[i,:], [len(X[i, -1].split(',')), 1]) for i in idx])

    for i in range(len(column_names)):
         plt.hist(X_signal[:,i], log=True, label="Signal")
         plt.hist(X_noise[:,i], log=True, alpha=0.7, label="Noise")
         plt.hist(tmp[:, i].astype(float), log=True, alpha=0.7, label="False positives")
         plt.legend()
         plt.xlabel(column_names[i])
         plt.savefig(plot_dir + column_names[i] +".png")
         plt.clf()

    scatter = ['netcc0', 'netcc2', 'Qveto', 'Lveto', 'rho0', 'lag', 'chirp1']
    for i in scatter:
        for j in scatter:
            if i > j:                                               
                plt.scatter(X[idx, column_numbers[i]].astype(float), 
                            X[idx, column_numbers[j]].astype(float),
                             c=[len(i.split(',')) for i in X[idx, -1]])
                plt.xlabel(i)
                plt.ylabel(j)
                plt.colorbar()
                plt.savefig(plot_dir + i + 'vs' + j +".png")
                plt.clf()
                 
     
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

