import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

import random
import numpy as np
import tensorflow as tf
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)



from options.options import BaseOptions
from utils.setup import check_if_there_is_uncommited_changes, setup_options
from models.model_factory import create_model
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import KFold
import gc


# check_if_there_is_uncommited_changes()

SAVE_DIR = PROJECT_ROOT + '/KaggleCompetition-GlomerularClassification/artifacts/'+ datetime.now().strftime("%Y%m%d_%H-%M-%S_")
opt = BaseOptions().parse()


for dataset_name in opt.datasets:

    print('%'*100)
    print(dataset_name)

    x = np.load(f'data/{dataset_name}/x.npy')
    y = tf.keras.utils.to_categorical(np.load(f'data/{dataset_name}/y.npy'))
    x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    if opt.training_mode == 'default':

        model = create_model(opt)
        model.set_dataset(dataset_name)
        model.set_save_dir(SAVE_DIR)

        print('%'*100)
    
        model.compile()
        model.train(x_train, y_train)
        model.test(x_test, y_test)
        model.save_predictions()
        model.save_model()
        model.save_metrics()
        model.save_plots()
        gc.collect()

    elif opt.training_mode == 'cross-validation': 

        print('%'*100)
        print(dataset_name)

        kfold = KFold(n_splits=opt.cross_validation_folds, shuffle=True)
        fold_no = 1

        for train, test in kfold.split(x, y):

            model = create_model(opt)
            model.set_dataset(dataset_name)
            model.set_save_dir(SAVE_DIR + 'CV_')

            model.compile()
            model.set_run(fold_no)
            model.train(x[train], y[train])
            model.test(x[test], y[test])
            model.save_predictions()
            model.save_model()
            model.save_metrics()
            model.save_plots()
            gc.collect()

            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
   
            fold_no = fold_no + 1

   