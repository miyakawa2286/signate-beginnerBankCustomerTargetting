import os
print(os.getcwd())
import sys

import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./my_modules/')
from utils import TARGET
from utils import read_dataset
from mls import get_evaluate
from mytorch import DEVICE, MyNormalizer, MyDataset, Net, my_mlp_trainer

CLEAN_LOG = True

EPOCH_SIZE = 5
BATCH_SIZE = 256
OPTIMIZER = 'adagrad'
OPTIMIZER_LR = 0.1


# define path
dpath_to_data = './data/raw/'
dpath_to_logs = './projects/z99-pytorchPractice/logs'
dpath_to_model = './projects/z99-pytorchPractice/models'

# load data
train, test, sub = read_dataset(dpath_to_data)
features = [col for col in train if col!=TARGET]

# encoding categorical features
# one hot encoding
cats = train.columns[train.dtypes=='category'].to_list()
ce_ohe = ce.OneHotEncoder(cols=cats, handle_unknown='value')
train = pd.concat([ce_ohe.fit_transform(train[features]),train[TARGET]],axis=1)
test = ce_ohe.transform(test[features])
enc_features = [col for col in train if col!=TARGET]

# split train into local train and val
X_train, X_val, y_train, y_val = train_test_split(
    train[enc_features],
    train[TARGET],
    test_size=0.2,
    random_state=2020,
    shuffle=True,
    stratify=train[TARGET],
)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}\nX_val: {X_val.shape}, y_val: {y_val.shape}')
# fit normalizer
nl = MyNormalizer()
nl.fit(X_train)

# make dataset and dataloader
train_dataset = MyDataset(X_train, y_train.to_list(), transform=nl)
val_dataset = MyDataset(X_val, y_val.to_list(), transform=nl)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# init model, optimizer, loss
net = Net(inp_size=len(enc_features), out_size=1).to(DEVICE)
if OPTIMIZER=='sgd':
    optimizer = optim.SGD(net.parameters(), lr=OPTIMIZER_LR)
elif OPTIMIZER=='adam':
    optimizer = optim.Adam(net.parameters(), lr=OPTIMIZER_LR)
elif OPTIMIZER=='rmsprop':
    optimizer = optim.RMSprop(net.parameters(), lr=OPTIMIZER_LR)
elif OPTIMIZER=='adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=OPTIMIZER_LR)
else:
    raise Exception(f'Not found {OPTIMIZER}')
criterion = nn.BCELoss()
# update path to seperately save logs
dpath_to_logs = os.path.join(dpath_to_logs,f'{OPTIMIZER}_lr{OPTIMIZER_LR}')
dpath_to_model = os.path.join(dpath_to_model,f'{OPTIMIZER}_lr{OPTIMIZER_LR}')
if not os.path.exists(dpath_to_model):
    os.makedirs(dpath_to_model, exist_ok=True)

# tensorboard log writer
if CLEAN_LOG:
    for sdir,_,files in os.walk(dpath_to_logs):
        if files:
            for f in files:
                os.remove(os.path.join(sdir,f))
train_writer = SummaryWriter(log_dir=os.path.join(dpath_to_logs,'train'))
val_writer = SummaryWriter(log_dir=os.path.join(dpath_to_logs,'val'))

# run training
my_mlp_trainer(
    epoch_size=EPOCH_SIZE,
    net=net,
    criterion=criterion,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    train_writer=train_writer,
    val_writer=val_writer,
    dpath_to_model=dpath_to_model,
    verbose=True,
)
