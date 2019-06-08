import csv
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext

import numpy as np

def load_data(file_path, cols):
    data = np.genfromtxt(file_path, delimiter=',', usecols=range(0,int(cols)))
    target = np.genfromtxt(file_path, delimiter=',', usecols=-1)
    #print('-------------- DATA ------------------------------------')
    #print(data)
    #print('-------------- TARGET ------------------------------------')
    #print(target)
    return data, target                 