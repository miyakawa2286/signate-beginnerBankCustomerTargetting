import os
print(os.getcwd())
import sys
import pickle
import codecs
import pprint

import torch.nn as nn
import torch.optim as optim

sys.path.append('./my_modules')
from utils import TARGET, read_dataset
from mls import my_cross_validation
from mytorch import DEVICE, Net

CLEAN_OUTPUT = True

CV_VER = 'v1'
FEATURE_VER = 'v1'
EPOCH_SIZE = 5
BATCH_SIZE = 256
OPTIMIZER = 'adam'
OPTIMIZER_LR = 0.1
CLF_THRESHOLD = 0.5


# define path
dpath_to_data = f'./data/processed/{FEATURE_VER}'
dpath_to_out = f'./projects/z99-pytorchPractice/crossvalidation/{CV_VER}'
# clean up output directory
if CLEAN_OUTPUT:
    for sdir,_,files in os.walk(dpath_to_out):
        if files:
            for f in files:
                os.remove(os.path.join(sdir,f))

# load data
train, test, sub = read_dataset(dpath_to_data)
features = [col for col in train if col!=TARGET]

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

# cross validation
# define config for cross validation
cv_conf = {
    'feature_ver': FEATURE_VER,
    'name': 'mlp',
    'training_params': {
        'epoch_size': EPOCH_SIZE,
        'batch_size': BATCH_SIZE,
        'net': net,
        'criterion': criterion,
        'optimizer': optimizer,
        'verbose': False,
    },
    'clf_threshold': CLF_THRESHOLD,
}
# run cross validation
print('[info] start cross validation')
cv_res = my_cross_validation(
    cv_conf,
    train,
    TARGET,
    features,
    dpath_to_out,
    verbose=True,
)
# write results
# score
cv_res['eval'].to_csv(os.path.join(dpath_to_out,'cv_eval.csv'))
# prediction
cv_res['pred'].to_csv(os.path.join(dpath_to_out,'cv_pred.csv'))
# config
with open(os.path.join(dpath_to_out,'config_serial.pkl'),'wb') as f:
    pickle.dump(cv_conf, f)
model_conf_string = pprint.pformat(cv_conf, indent=1)
print(model_conf_string, file=codecs.open(os.path.join(dpath_to_out,'config_string.txt'), 'w', 'utf-8'))
