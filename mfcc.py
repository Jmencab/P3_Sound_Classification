import re
import csv
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import numpy
from tqdm import tqdm

def mfcc(S):
  return librosa.feature.mfcc(S, n_mfcc=100).flatten()

f_handle = file('mfcc.csv', 'a')
for x in range(0,6325,25):
	f_handle = file('mfcc.csv', 'a')
	data = pd.read_csv('train.csv', skiprows=x, nrows=25)
	train = data.values
	X_train, y_train = train[:, :-1], train[:, -1]
	print "X and Y shape", X_train.shape, y_train.shape
	out = np.apply_along_axis(mfcc, 1, X_train)
	np.savetxt(f_handle, out, delimiter=",")
f_handle.close()
