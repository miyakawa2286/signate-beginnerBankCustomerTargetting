import os
print(os.getcwd())
import sys

sys.path.append('./my_modules/')
from utils import TARGET, LOCAL_TRAIN_RATIO
from utils import read_dataset
from mls import agg_cv_models
from utils import make_submission

dpath_to_cv_models = './projects/b01-baselineLightGBM/res/cv_models/'
dpath_to_output = './projects/b01-baselineLightGBM/res/'

# load dataset
train,test,sub = read_dataset('./data/')
features = [c for c in train.columns if c!=TARGET]

# agg cv models
res = agg_cv_models(train,
                    test,
                    features,
                    TARGET,
                    dpath_to_cv_models,
                    )

# write predict
make_submission(base_df=sub,
                pred=res['agg_pred'],
                dpath_to_output=dpath_to_output,
                filename='agg_cv_models_pred.csv',
                )
