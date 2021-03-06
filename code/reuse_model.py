import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

model_loc = sys.argv[1]

saver = tf.train.import_meta_graph(model_loc + '.meta')
sess = tf.Session()
saver.restore(sess, model_loc)
graph = tf.get_default_graph()

x = graph.get_tensor_by_name('model/input:0')
y = graph.get_tensor_by_name('model/label:0')
output = graph.get_tensor_by_name('model/pred:0')
y_ = graph.get_tensor_by_name('model/prob/Sigmoid:0')
correct_prediction = graph.get_tensor_by_name('model/cp:0')

# data1 = sio.loadmat('data/R3_cWB_BBH_BKG_with_rho_g6cor.mat')['data']
data1 = sio.loadmat('data/R3_cWB_BKG_with_qveto_gp3_dp2.mat')['data']
data1 = data1[~np.isnan(data1).any(axis=1)]
# cols_used = [2, 3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19]
# cols_used = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
cols_used = [2, 3, 4, 5, 6, 7, 8, 10, 11, 16, 17, 18, 19]

pred1 = sess.run(output, feed_dict={x:data1[:, cols_used]}).reshape(-1)
prob1 = sess.run(y_, feed_dict={x:data1[:, cols_used]}).reshape(-1)

far = len(np.where(pred1!=0)[0]) * 100 / len(pred1)
print('FAR : ', far, '%')
sess.close()
