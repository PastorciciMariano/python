import random
import os
import numpy as np
import tensorflow as tf
import keras as kr
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

os.environ['http_proxy']= 'http://proxy-chain.intel.com:912'
os.environ['https_proxy']= 'http://proxy-chain.intel.com:912'
filepath = kr.utils.get_file('text.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]

characters = sorted(set(text))


char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

TBD