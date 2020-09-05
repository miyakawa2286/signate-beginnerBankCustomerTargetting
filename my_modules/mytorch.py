import os

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score #, accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def __init__(
        self,
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


def my_mlp_trainer(
    epoch_size: int,
    net,
    criterion,
    optimizer,
    train_dataloader,
    val_dataloader,
    train_writer,
    val_writer,
    dpath_to_model: str,
    verbose: bool = False,
    ):
    # define training parameters
    print_step = 10
    min_val_loss = np.inf
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
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
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
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
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
            if verbose and i % print_step == print_step-1:    # print every 100 mini-batches
                print('[%d, %4d]: train loss: %.3f, val loss: %0.3f' %(epoch + 1, i + 1, running_train_loss/print_step, running_val_loss/print_step))
                running_train_loss = running_val_loss = 0.0
    train_writer.close()
    val_writer.close()


if __name__=='__main__':
    print(f'device: {DEVICE}')
