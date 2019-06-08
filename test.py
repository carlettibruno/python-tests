import csv
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import numpy as np
from io import StringIO

data = u"1 2 4 4 3\n4 5 11 32 6"
test = 4
data = np.genfromtxt(StringIO(data), usecols=-1)
print(data)