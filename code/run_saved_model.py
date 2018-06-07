import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

model_loc = sys.argv[1]
saver = tf.train.import_meta_graph(model_loc + '.meta')
with tf.Session() as sess:
    saver.restore(sess, model_loc)
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('Placeholder:0')
    y = graph.get_tensor_by_name('Placeholder_1:0')
    output = graph.get_tensor_by_name('Cast_1:0')
    data1 = sio.loadmat('R3_cWB_BBH_BKG_with_rho_g6cor.mat')['data']
    data1 = data1[~np.isnan(data1).any(axis=1)]
    cols_used = [2, 3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19]
    pred = sess.run(output, feed_dict={x:data1[:, cols_used]}).reshape(-1)
    y_ = graph.get_tensor_by_name('dense_1/Sigmoid:0')
    prob = sess.run(y_, feed_dict={x:data1[:, cols_used]}).reshape(-1)
    import pdb; pdb.set_trace()
    print('False Alarm Rate: ', len(np.where(pred!=0)[0]) * 100 / len(pred), '%')

