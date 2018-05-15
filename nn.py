import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


input_file = "ToHarish/params.csv"

cols = pd.read_csv(input_file, nrows=1).columns
n_params = len(cols) - 2

df = pd.read_csv(input_file, usecols=cols[1:], index_col=False)
df = shuffle(df).reset_index(drop=True)

df_train = df.loc[:int(len(df)/2)-1].reset_index(drop=True)
df_test = df.loc[int(len(df)/2):].reset_index(drop=True)

x_test = df_test[cols[1:-1]].as_matrix()
y_test = df_test[cols[-1]].as_matrix()
x_train = df_train[cols[1:-1]].as_matrix()
y_train = df_train[cols[-1]].as_matrix()

x = tf.placeholder(dtype = tf.float32, shape = [None, n_params])
y = tf.placeholder(dtype = tf.int32, shape = [None])
