import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


testdata = "data/12params_test_data.csv"
traindata = "data/12params_train_data.csv"

df_train = pd.read_csv(traindata)
df_train_weak = df_train.loc[df_train['snr0'] < 200].reset_index(drop=True)
df_test = pd.read_csv(testdata)
df_test_weak = df_test.loc[df_test['snr0'] < 200].reset_index(drop=True)
cols = list(pd.read_csv(traindata, nrows=1).columns)[1:]
# cols = cols[:6] + cols[10:]
# cols = ['netED', 'norm', 'rho0', 'rho1', 'duration0', 'duration1', 'freq0', 'freq1','snr0','snr1','netcc0', 'netcc1', 'label']
# cols = ['freq0', 'rho1', 'label']
n_params = len(cols)-1

x_test = df_test_weak[cols[:-1]].as_matrix()
y_test = df_test_weak[cols[-1]].as_matrix().astype(int)
x_train = df_train_weak[cols[:-1]].as_matrix()
y_train = df_train_weak[cols[-1]].as_matrix().astype(int)

learning_rate = 0.001
epochs = 100000
hidden_layer_size = 30

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
print("Prediction accuracy on test data: ", 100*acc, "%")

false_positives = df_test_weak.as_matrix()[np.where((pred - y_test) == 1)[0]][:, 0]
false_negatives = df_test_weak.as_matrix()[np.where((pred - y_test) == -1)[0]][:, 0]
fp = df_test_weak.loc[df_test_weak['filename'].isin(false_positives)].reset_index(drop=True)
fn = df_test_weak.loc[df_test_weak['filename'].isin(false_negatives)].reset_index(drop=True)

signal_test_weak = df_test_weak.loc[df_test_weak['label'] == 1].reset_index(drop=True)
noise_test_weak = df_test_weak.loc[df_test_weak['label'] == 0].reset_index(drop=True)
signal = df_test_weak.loc[df_test_weak['label'] == 1].append(df_train_weak.loc[df_train_weak['label'] == 1])
signal = signal.reset_index(drop=True)
noise = df_test_weak.loc[df_test_weak['label'] == 0].append(df_train_weak.loc[df_train_weak['label'] == 0])
noise = noise.reset_index(drop=True)

signal_all = df_test.loc[df_test['label'] == 1].append(df_train.loc[df_train['label'] == 1])
signal_all = signal_all.reset_index(drop=True)
noise_all = df_test.loc[df_test['label'] == 0].append(df_train.loc[df_train['label'] == 0])
noise_all = noise_all.reset_index(drop=True)


save_path = saver.save(sess, "checkpoints/model.ckpt")
print("Model saved in path: %s" % save_path)

opt = sess.run(y_, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)
cp = sess.run(correct_prediction, feed_dict={x:x_test, y:y_test.reshape(-1,1)}).reshape(-1)

# sess.close()
savedir = 'results/'

plt.hist(fp['rho1'], label="False Postives")
plt.hist(fn['rho1'], alpha=0.5, label="False Negatives")
plt.xlabel("$\\rho$")
plt.legend()
plt.savefig(savedir + 'fpfn.png')
plt.clf()

plt.hist((signal_test_weak['rho1']), label="Signal")
plt.hist((noise_test_weak['rho1']), alpha=0.7, label="Noise")
plt.xlabel("$\\rho$")
plt.legend()
plt.savefig(savedir + 'rho.png')
plt.clf()


# plt.matshow(w, cmap=plt.cm.Spectral_r, interpolation='none')
# plt.colorbar()
# plt.show()

# plt.matshow(w2, cmap=plt.cm.Spectral_r, interpolation='none')
# plt.colorbar()
# plt.show()
