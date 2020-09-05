import os
print(os.getcwd())
import sys

import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score#, accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./my_modules/')
from utils import TARGET
from utils import read_dataset
from mls import get_evaluate

CLEAN_LOG = True

EPOCH_SIZE = 5
BATCH_SIZE = 256
OPTIMIZER = 'adam'
OPTIMIZER_LR = 0.01

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
net = Net(inp_size=len(enc_features), out_size=1).to(device)
if OPTIMIZER=='sgd':
    optimizer = optim.SGD(net.parameters(), lr=OPTIMIZER_LR)
elif OPTIMIZER=='adam':
    optimizer = optim.Adam(net.parameters(), lr=OPTIMIZER_LR)
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

# define training parameters
print_step = 10
min_val_loss = np.inf
#validation_step = int(len(train_dataloader)*0.1)
# iterate over minibatchs
for epoch in range(EPOCH_SIZE):  # loop over the dataset multiple times
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
        # write logs
        # loss
        train_writer.add_scalar(
            "Loss", 
            loss.item(),
            epoch*len(train_dataloader)+i,
            )
        # auc
        if not all(labels==0):
            train_writer.add_scalar(
                "AUC", 
                roc_auc_score(labels.to('cpu').detach().numpy(), outputs.to('cpu').detach().numpy()),
                epoch*len(train_dataloader)+i,
                )
        # write network graph
        if i==0:
            train_writer.add_graph(net, inputs)
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
            val_loss = loss.item()
            running_val_loss += val_loss
            # write parameters
            if val_loss < min_val_loss:
                torch.save(net.state_dict(), os.path.join(dpath_to_model,'model.pt'))
                min_val_loss = val_loss 
            # write logs
            # loss
            val_writer.add_scalar(
                "Loss", 
                loss.item(),
                epoch*len(train_dataloader)+i,
                )
            # auc
            if not all(labels==0):
                val_writer.add_scalar(
                    "AUC", 
                    roc_auc_score(labels.to('cpu').detach().numpy(), outputs.to('cpu').detach().numpy()),
                    epoch*len(train_dataloader)+i,
                    )
        ## print statistics
        if i % print_step == print_step-1:    # print every 100 mini-batches
            print('[%d, %4d]: train loss: %.3f, val loss: %0.3f' %(epoch + 1, i + 1, running_train_loss/print_step, running_val_loss/print_step))
            running_train_loss = running_val_loss = 0.0
train_writer.close()
val_writer.close()
