import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()


    def initialize(self):      
        self.parser.add_argument('--model_name', type=str, default='AlexNet', help='Nome do modelo para treinar')
        self.parser.add_argument('--training_mode', type=str, default='default', help='Tipos de pipelines de treinos(default, cross-validation)')
        self.parser.add_argument('--datasets', nargs='*',default=['dataset_image_100x100', 'dataset_image_200x200'],type=str, help='Lista de datasets para treinar')
        self.parser.add_argument('--labels', type=list, default=['Hiperceluraridade', 'Normal'],help='labels das classes target.')
        self.parser.add_argument('--channels', type=int, default=3, help='Quantidade de canais de cada dataset')
        self.parser.add_argument('--num_classes', type=int, default=2, help='Quantidade de canais de cada dataset')
        self.parser.add_argument('--image_size', type=int, default=100, help='Qauntidade de pontos amostrais padão de cada seguimento de sinal')
        self.parser.add_argument('--epochs', type=int, default=200, help='Epocas para treinamento das redes')
        self.parser.add_argument('--batch_size', type=int, default=25, help='Tamanho do batch para treinamento da rede')
        self.parser.add_argument('--cross_validation_folds', type=int, default=10, help='Quantidade de folds do crossvalidation')
        self.parser.add_argument('--data_augumentation', type=bool, default=False, help='Quantidade de folds do crossvalidation')
    
    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt 


class OptionsObject:
    def __init__(self) -> None:
        self.training_mode = None
        self.datasets = None
        self.labels = None
        self.model_name = None
        self.channels = 200
        self.num_classes = None
        self.image_size = None
        self.epochs = None
        self.batch_size = None
        self.cross_validation_folds = None