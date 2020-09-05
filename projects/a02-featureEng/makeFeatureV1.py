import os 
import sys 

import pandas as pd 
import category_encoders as ce

sys.path.append('./my_modules')
from utils import TARGET, read_dataset


# define path
dpath_to_data = './data/raw/'
dpath_to_out = './data/processed/v1'

# load data
train, test, sub = read_dataset(dpath_to_data)
features = [col for col in train if col!=TARGET]

# encoding categorical features with one hot encoding
cats = train.columns[train.dtypes=='category'].to_list()
ce_ohe = ce.OneHotEncoder(cols=cats, handle_unknown='value')
train = pd.concat([ce_ohe.fit_transform(train[features]),train[TARGET]],axis=1)
test = ce_ohe.transform(test[features])
enc_features = [col for col in train if col!=TARGET]
print(f'train shape: {train[enc_features].shape}')
print(f'test shape: {test.shape}')

# write result
if not os.path.exists(dpath_to_out):
    os.makedirs(dpath_to_out,exist_ok=True)
train.to_csv(os.path.join(dpath_to_out,'train.csv'))
test.to_csv(os.path.join(dpath_to_out,'test.csv'))
