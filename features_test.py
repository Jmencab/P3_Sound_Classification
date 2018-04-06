import re
import numpy as np
import pandas as pd
import string
import librosa
import librosa.display
import torch

def extract_features(row):
  X, sample_rate = row, 22050
  stft = np.abs(librosa.stft(X))
  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
  chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
  mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
  contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
  tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
  features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
  return features#[np.newaxis, :]

data = pd.read_csv('test.csv', header=None)#, skiprows=900, nrows=100)
test = data.values
# print "Train shape", train.shape

X_train = test[:, 1:]
features = np.apply_along_axis(extract_features, 1, X_train)
# print features.shape
np.savetxt("test_features.csv", features, delimiter=",")
