
# from xgboost import XGBClassifier
from .AlexNet import AlexNet
from .LeNet import LeNet
from .ResNet import ResNet50
from .VGG import VGG16
from .Inception import Inception
from .Xception import Xception

from .base_model import BaseModel


class AlexNet_(BaseModel):
    def __init__(self, opt):
        model = AlexNet(input_shape=(
            opt.image_size, opt.image_size, 3), num_classes=opt.num_classes)
        super().__init__(model, opt)


class LeNet_(BaseModel):
    def __init__(self, opt):
        model = LeNet(input_shape=(opt.image_size, opt.image_size,
                      3), nb_classes=opt.num_classes)
        super().__init__(model, opt)


class ResNet50_(BaseModel):
    def __init__(self, opt):
        model = ResNet50(input_shape=(
            opt.image_size, opt.image_size, 3), num_classes=opt.num_classes)
        super().__init__(model, opt)


class VGG16_(BaseModel):
    def __init__(self, opt):
        model = VGG16(input_shape=(opt.image_size, opt.image_size,
                      3), num_classes=opt.num_classes)
        super().__init__(model, opt)


class VGG16_(BaseModel):
    def __init__(self, opt):
        model = VGG16(input_shape=(opt.image_size, opt.image_size,
                      3), num_classes=opt.num_classes)
        super().__init__(model, opt)


class Inception_(BaseModel):
    def __init__(self, opt):
        model = Inception(input_shape=(
            opt.image_size, opt.image_size, 3), num_classes=opt.num_classes)
        super().__init__(model, opt)


class Xception_(BaseModel):
    def __init__(self, opt):
        model = Xception(input_shape=(
            opt.image_size, opt.image_size, 3), num_classes=opt.num_classes)
        super().__init__(model, opt)
