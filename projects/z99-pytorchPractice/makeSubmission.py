import os
print(os.getcwd())
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./my_modules')
from utils import TARGET, read_dataset, make_submission
from mytorch import DEVICE, MyNormalizer, Net, my_mlp_trainer

CLEAN_LOG = True

EPOCH_SIZE = 5
BATCH_SIZE = 256
OPTIMIZER = 'adam'
OPTIMIZER_LR = 0.1
CLF_THRESHOLD = 0.5

# set seed
np.random.seed(2020)

# define path
dpath_to_data = './data/processed/v1'
dpath_to_out = './projects/z99-pytorchPractice/submission'
dpath_to_logs = os.path.join(dpath_to_out, 'logs')

# load data
train, test, ss = read_dataset(dpath_to_data)
features = [col for col in train if col!=TARGET]

# get local train and validation data
local_train, local_val = train_test_split(
    train,
    test_size=0.1,
    random_state=2020,
    shuffle=True,
    stratify=train[TARGET],
)
print(f'local_train: {local_train.shape}, local_test: {local_val.shape}')

# init
# model
net = Net(inp_size=len(features), out_size=1).to(DEVICE)
# optimizer
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
#criterion
criterion = nn.BCELoss()
# normaizer
nm = MyNormalizer()
nm.fit(local_train[features])
# update path to seperately save logs
# logs directory is automatically created by tensorboard writer
dpath_to_logs = os.path.join(dpath_to_logs,f'{OPTIMIZER}_lr{OPTIMIZER_LR}')

# tensorboard log writer
# clean up log directory
if CLEAN_LOG:
    for sdir,_,files in os.walk(dpath_to_logs):
        if files:
            for f in files:
                os.remove(os.path.join(sdir,f))
train_writer = SummaryWriter(log_dir=os.path.join(dpath_to_logs,'train'))
val_writer = SummaryWriter(log_dir=os.path.join(dpath_to_logs,'val'))

# run training
net = my_mlp_trainer(
    epoch_size=EPOCH_SIZE,
    batch_size=BATCH_SIZE,
    net=net,
    criterion=criterion,
    optimizer=optimizer,
    normalizer=nm,
    X_train=local_train[features],
    y_train=local_train[TARGET],
    X_val=local_val[features],
    y_val=local_val[TARGET],
    fpath_to_model_state_dict=os.path.join(dpath_to_out,'model_state_dict.pt'),
    train_writer=train_writer,
    val_writer=val_writer,
    return_model=True,
    verbose=True,
)

# get prediction for test data
net.eval()
with torch.no_grad():
    val_batch = nm.transform(test[features])
    val_batch = torch.from_numpy(val_batch.values).float().to(DEVICE)
    predict_proba = net(val_batch)
    predict_proba = predict_proba.to('cpu').detach().numpy()
predict = [0 if i < CLF_THRESHOLD else 1 for i in predict_proba]
# make submission file
make_submission(
    ss,
    predict_proba,
    dpath_to_out,
    filename=None,
)
