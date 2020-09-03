import os
print(os.getcwd())
import sys

import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./my_modules/')
from utils import TARGET
from utils import read_dataset

CLEAN_LOG = True


class MyNormalizer:
    '''
    x = (x-mean)/std
    
    '''
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self,df):
        self.mean = df.mean()
        self.std = df.std()
    
    def transform(self,df):
        return (df-self.mean)/self.std
    
    def inverse_transform(self,df):
        return df*self.std + self.mean


class MyDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 label: list,
                 transform,
                 ):
        self.data = data.reset_index(drop=True)
        self.label = torch.FloatTensor([float(t) for t in label])
        self.label = self.label.view(-1,1)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        out_data = self.transform.transform(self.data.iloc[idx])
        out_data = torch.from_numpy(out_data.values).float()
        return out_data, self.label[idx]


class Net(nn.Module):
    def __init__(self,inp_size,out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inp_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, out_size)
    
    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x

# load data
dpath_to_data = './data/raw/'
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
    batch_size=64,
    shuffle=True,
    num_workers=0,
    )
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=64,
    shuffle=True,
    num_workers=0,
    )

# init model, optimizer, loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
net = Net(inp_size=len(enc_features), out_size=1).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.BCELoss()

# tensorboard log writer
dpath_to_logs = './projects/z99-pytorchPractice/logs/'
if CLEAN_LOG:
    for sdir,_,files in os.walk(dpath_to_logs):
        if files:
            for f in files:
                os.remove(os.path.join(sdir,f))
train_loss_writer = SummaryWriter(log_dir=os.path.join(dpath_to_logs,'train-loss'))
val_loss_writer = SummaryWriter(log_dir=os.path.join(dpath_to_logs,'val-loss'))
# define training parameters
epoch_size = 25
print_step = 10
#validation_step = int(len(train_dataloader)*0.1)
# iterate over minibatchs
for epoch in range(epoch_size):  # loop over the dataset multiple times
    running_train_loss = 0.0
    running_val_loss = 0.0
    for i, data in enumerate(train_dataloader,0):
        ## training
        # set model to train mode
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # set on gpu
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        # log the training loss
        train_loss_writer.add_scalar(
            "loss", 
            loss.item(),
            epoch*len(train_dataloader)+i,
            )        
        ## randomly get one minibatch from val_dataloader
        ## and eval with them
        with torch.no_grad():
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = next(iter(val_dataloader))
            # set on gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward + evaluation
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            # log the val loss
            val_loss_writer.add_scalar(
                "loss", 
                loss.item(),
                epoch*len(train_dataloader)+i,
                )
        ## print statistics
        if i % print_step == print_step-1:    # print every 100 mini-batches
            print('[%d, %4d]: train loss: %.3f, val loss: %0.3f' %(epoch + 1, i + 1, running_train_loss/print_step, running_val_loss/print_step))
            running_train_loss = running_val_loss = 0.0
        
print('Finished Training')
train_loss_writer.close()
val_loss_writer.close()
