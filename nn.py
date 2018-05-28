import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scipy.io as sio


# testdata = "7params_test_data.csv"
# traindata = "7params_train_data.csv"
column_names = {0:'lag', 1:'slag', 2:'rho0', 3:'rho1', 4:'netcc0', 5:'netcc2', 6:'penalty', 7:'netED', 8:'likelihood', 9:'duration', 10:'frequency1', 11:'frequency2', 12:'Qveto', 13:'Lveto', 14:'chirp1', 15:'chirp2', 16:'strain', 17:'hrssL', 18:'hrssH', 19:'SNR1'}
column_numbers = dict((v,k) for k,v in column_names.items())

# df_train = pd.read_csv(traindata)
# df_train_weak = df_train.loc[df_train['snr0'] < 200].reset_index(drop=True)
# df_test = pd.read_csv(testdata)
# df_test_weak = df_test.loc[df_test['snr0'] < 200].reset_index(drop=True)
# cols = list(pd.read_csv(traindata, nrows=1).columns)[1:]
# cols = cols[:6] + cols[10:]
# cols = ['netED', 'norm', 'rho0', 'rho1', 'duration0', 'duration1', 'freq0', 'freq1','snr0','snr1','netcc0', 'netcc1', 'label']
# cols = ['freq0', 'rho1', 'label']
# n_params = len(cols)-1

X_signal = sio.loadmat('data/simulation_R3_space.mat')['data']
X_signal = np.append(X_signal, np.ones((len(X_signal), 1)), 1)
X_noise = sio.loadmat('data/R3_cWB_BKG_with_qveto_gp3_dp2.mat')['data']
X_noise = np.append(X_noise, np.zeros((len(X_noise), 1)), 1)
X = shuffle(np.append(X_signal, X_noise, 0))
X = X[~np.isnan(X).any(axis=1)]
# X_noise = np.append(sio.loadmat('data/R3_cWB_BBH_BKG_with_rho_g6_netcc_gp7_qveto_gp3_twin_p2.mat')['data'],
#                     sio.loadmat('data/R3_cWB_BKG_with_qveto_gp3_dp2.mat')['data'], 0)
n_params = X.shape[1] - 1
x_test = X[:int(len(X)/2), :-1]
y_test = X[:int(len(X)/2), -1]
x_train = X[int(len(X)/2):, :-1]
y_train = X[int(len(X)/2):, -1]
# x_test = df_test_weak[cols[:-1]].as_matrix()
# y_test = df_test_weak[cols[-1]].as_matrix().astype(int)
# x_train = df_train_weak[cols[:-1]].as_matrix()
# y_train = df_train_weak[cols[-1]].as_matrix().astype(int)

x = tf.placeholder(dtype = tf.float32, shape = [None, n_params])
y = tf.placeholder(dtype = tf.int32, shape = [None, 1])

learning_rate = 0.001
epochs = 100000
batch_size = 100
hidden_layer_size = 22

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
for i in range(epochs):
    _, c = sess.run([optimiser, loss], 
                 feed_dict={x: x_train, y: y_train.reshape(-1,1)})
    if (i%1000 == 0):
        acc = sess.run(accuracy,
                       feed_dict={x: x_train, y: y_train.reshape(-1,1)})
        print("Loss = ", c, ", Prediction accuracy = ", 100*acc, "%")
# import pdb; pdb.set_trace()
acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test.reshape(-1,1)})
pred = sess.run(output, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
prob = sess.run(y_, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
print("Prediction accuracy on test data: ", 100*acc, "%")

# false_positives = df_test_weak.as_matrix()[np.where((pred - y_test) == 1)[0]][:, 0]
# false_negatives = df_test_weak.as_matrix()[np.where((pred - y_test) == -1)[0]][:, 0]
# fp = df_test_weak.loc[df_test_weak['filename'].isin(false_positives)].reset_index(drop=True)
# fn = df_test_weak.loc[df_test_weak['filename'].isin(false_negatives)].reset_index(drop=True)
test_signal = x_test[np.where(y_test == 1)]
test_noise = x_test[np.where(y_test == 0)]

save_path = saver.save(sess, "checkpoints/model.ckpt")
print("Model saved in path: %s" % save_path)

weights = tf.get_default_graph().get_tensor_by_name(hl1.name.split('/')[0]+'/kernel:0')
weights2 = tf.get_default_graph().get_tensor_by_name(y_.name.split('/')[0]+'/kernel:0')
w = sess.run(weights)
w2 = sess.run(weights2)

opt = sess.run(y_, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
cp = sess.run(correct_prediction, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
plt.hist(opt[np.where(cp == 0)], 50,log=True); plt.show()

sess.close()

fp = x_test[np.where((pred - y_test) == 1)[0], :]
fn = x_test[np.where((pred - y_test) == -1)[0], :]
if(acc != 1):
    plt.hist(fp[:, column_numbers['rho1']], label="False Postives")
    plt.hist(fn[:, column_numbers['rho1']], alpha=0.5, label="False Negatives")
    plt.xlabel("$\\rho1$")
    plt.legend()
    plt.show()

plt.hist(prob[np.where(cp==1)], 30, log=True,label="Correct predictions")
plt.hist(prob[np.where(cp==0)], 30, alpha=0.5, log=True, label="Wrong predictions")
plt.xlabel("Confidence")
plt.legend()
plt.show()

plt.hist((test_signal[:, column_numbers['rho1']]), label="Signal")
plt.hist((test_noise[:, column_numbers['rho1']]), alpha=0.7, label="Noise")
plt.xlabel("$\\rho$")
plt.legend()
plt.show()

# plt.matshow(w, cmap=plt.cm.Spectral_r, interpolation='none')
# plt.colorbar()
# plt.show()

# plt.matshow(w2, cmap=plt.cm.Spectral_r, interpolation='none')
# plt.colorbar()
# plt.show()
