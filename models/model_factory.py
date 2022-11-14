from .models_config import (
    AlexNet_,
    LeNet_,
    ResNet50_,
    VGG16_,
    Inception_,
    Xception_
    )


def create_model(opt):
    print(opt.model_name)
    if opt.model_name == 'AlexNet':
        model = AlexNet_(opt)

    elif opt.model_name == 'LeNet':
        model = LeNet_(opt)

    elif opt.model_name == 'ResNet50':
        model = ResNet50_(opt)

    elif opt.model_name == 'VGG16':
        model = VGG16_(opt)

    elif opt.model_name == 'Inception':
        model = Inception_(opt)

    elif opt.model_name == 'Xception':
        model = Xception_(opt)

    else:
        raise TypeError(f'There is no model called {opt.model_name}')

    return model

