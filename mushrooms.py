import pandas as pd
from dataset import make_dataframe
import numpy as np
import tensorflow as tf
frame = make_dataframe()
print(frame.head())
labels = np.array(frame[['class_edible', 'class_poisonous']])
del(frame['class_edible'])
del(frame['class_poisonous'])
data = np.array(frame)
print(labels[0:2])
print(data[0:2])