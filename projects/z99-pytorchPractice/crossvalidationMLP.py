import os
print(os.getcwd())
import sys

sys.path.append('./my_modules')
from mytorch import MyNormalizer, MyDataset, Net
