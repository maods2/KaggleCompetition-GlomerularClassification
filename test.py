import os
from datetime import datetime
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

from models.AlexNet import AlexNet
from models.LeNet import LeNet
from models.ResNet import ResNet50
from models.VGG import VGG16
from models.Inception import Inception
from models.Xception import Xception

from models.base_model import BaseModel, OptionsObject
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

SAVE_DIR = PROJECT_ROOT + '/KaggleCompetition-GlomerularClassification/artifacts/'+ datetime.now().strftime("%Y%m%d_%H-%M-%S_")





x = np.load('data/dataset_image_100x100/x.npy')
y = tf.keras.utils.to_categorical(np.load('data/dataset_image_100x100/y.npy'))

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(x.shape)
print(y.shape)


obj = OptionsObject('AlexNet',['Hipercelularidade','Normal' ])
obj.epochs = 20
model = Xception_(obj)

model.set_dataset('100x100')
model.set_save_dir(SAVE_DIR)
model.compile()
model.train(x_train,y_train)
model.test(x_test, y_test)
model.save_predictions()
model.save_model()
model.save_metrics()
model.save_plots()