import os
print(os.getcwd())
import sys

import numpy as np
import lightgbm as lgb

sys.path.append('./my_modules/')
from utils import read_dataset

train,test,sub = read_dataset('./data/')

print(1)
