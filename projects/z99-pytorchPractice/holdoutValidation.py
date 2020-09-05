import os
print(os.getcwd())
import sys
import pickle
import codecs
import pprint

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./my_modules/')
from utils import TARGET, read_dataset
from mls import get_evaluate
from mytorch import DEVICE, MyNormalizer, Net, my_mlp_trainer

CLEAN_LOG = True

HO_VER = 'v1'
FEATURE_VER = 'v1'
EPOCH_SIZE = 5
BATCH_SIZE = 256
OPTIMIZER = 'adam'
OPTIMIZER_LR = 0.1
CLF_THRESHOLD = 0.5,

# set seed
np.random.seed(2020)

# define path
dpath_to_data = f'./data/processed/{FEATURE_VER}'
dpath_to_out = f'./projects/z99-pytorchPractice/holdout/{HO_VER}'
dpath_to_logs = os.path.join(dpath_to_out, 'logs')
dpath_to_model = os.path.join(dpath_to_out, 'models')

# load data
train, test, sub = read_dataset(dpath_to_data)
features = [col for col in train if col!=TARGET]

# split into ho_train and ho_test
ho_train, ho_test = train_test_split(
    train,
    test_size=0.1,
    random_state=2020,
    shuffle=True,
    stratify=train[TARGET],
)
# get local train and validation data
local_train, local_val = train_test_split(
    ho_train,
    test_size=0.1,
    random_state=2020,
    shuffle=True,
    stratify=ho_train[TARGET],
)
print(f'local_train: {local_train.shape}, local_test: {local_val.shape}, ho_test: {ho_test.shape}')

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
dpath_to_model = os.path.join(dpath_to_model,f'{OPTIMIZER}_lr{OPTIMIZER_LR}')
if not os.path.exists(dpath_to_model):
    os.makedirs(dpath_to_model, exist_ok=True)

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
ho_conf = {
    'name': 'mlp',
    'training_params': {
        'epoch_size': EPOCH_SIZE,
        'batch_size': BATCH_SIZE,
        'net': net,
        'criterion': criterion,
        'optimizer': optimizer,
        'normalizer': nm,
    },
    'clf_threshold': CLF_THRESHOLD,
}
net = my_mlp_trainer(
    X_train=local_train[features],
    y_train=local_train[TARGET],
    X_val=local_val[features],
    y_val=local_val[TARGET],
    fpath_to_model_state_dict=os.path.join(dpath_to_model,'model_state_dict.pt'),
    train_writer=train_writer,
    val_writer=val_writer,
    verbose=True,
    **ho_conf['training_params'],
)

# prediction and evaluation for holdout validation data
net.eval()
with torch.no_grad():
    val_batch = nm.transform(ho_test[features])
    val_batch = torch.from_numpy(val_batch.values).float().to(DEVICE)
    predict_proba = net(val_batch)
    predict_proba = predict_proba.to('cpu').detach().numpy()
    auc, precision, recall = get_evaluate(ho_test[TARGET], predict_proba, CLF_THRESHOLD)
print('holdout validation result:')
print('auc: %.5f, precision: %.5f, recall: %.5f' %(auc, precision, recall))
# write result
# scores
with open(os.path.join(dpath_to_out,'scores.txt'), 'a') as f:
    f.write(f'[{OPTIMIZER}_lr{OPTIMIZER_LR}]\n')
    f.write('auc: %.5f, precision: %.5f, recall: %.5f\n' %(auc, precision, recall))
# config
with open(os.path.join(dpath_to_out,'config_serial.pkl'),'wb') as f:
    pickle.dump(ho_conf,f)
model_conf_string = pprint.pformat(ho_conf, indent=1)
print(model_conf_string, file=codecs.open(os.path.join(dpath_to_out,'config_string.txt'), 'w', 'utf-8'))
