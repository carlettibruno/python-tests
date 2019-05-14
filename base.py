#import os
import csv
#import sys
#import shutil
#from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
#import hashlib

import numpy as np

def load_data(module_path, data_file_name):
    """Loads data from module_path/data/data_file_name.
    Parameters
    ----------
    module_path : string
        The module path.
    data_file_name : string
        Name of csv file to be loaded from
        module_path/data/data_file_name. For example 'wine_data.csv'.
    Returns
    -------
    data : Numpy array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.
    target : Numpy array
        A 1D array holding target variables for all the samples in `data.
        For example target[0] is the target varible for data[0].
    target_names : Numpy array
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.
    """

    data = np.genfromtxt(join(module_path, '/Users/brunocarletti/Downloads/', data_file_name), delimiter=',', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24))
    target = np.genfromtxt(join(module_path, '/Users/brunocarletti/Downloads/', data_file_name), delimiter=',', usecols=(25))
    #print('-------------- DATA ------------------------------------')
    #print(data)

    #print('-------------- TARGET ------------------------------------')
    #print(target)

    return data, target                 