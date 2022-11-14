from sklearn import metrics
import json
import numpy as np
from utils.plots import draw_confusion_matrix, draw_learning_curves, draw_roc_curves
import sys
import os
import pathlib
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from subprocess import Popen, PIPE
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# class OptionsObject:
#     def __init__(self, model_name, labels: list[str]) -> None:
#         self.training_mode = 'default'
#         self.epochs = 200
#         self.batch_size = 35
#         self.model_name = model_name
#         self.labels = labels
#         self.image_size = None
#         self.num_classes = None


class BaseModel:
    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
        self.y_pred = None
        self.y_test = None
        self.pred_bool = None
        self.x_test = None
        self.subject = None
        self.run = None
        self.dataset_name = None
        self.hist = None
        self.artifact_folder = None
        self.probs = None
        self.optmizer_setup = None
        self.metrics = dict()

    def train(self, x_train, y_train):
        # earlystopping = EarlyStopping(
        #     monitor='val_loss',  # set monitor metrics
        #     patience=10,  # number of epochs to stop training
        #     restore_best_weights=True,  # set if use best weights or last weights
        # )

        if self.opt.training_mode == 'cross-validation':

            if self.opt.data_augumentation:
                train_datagen = ImageDataGenerator(rotation_range=5,  # rotation
                                                   width_shift_range=0.2,  # horizontal shift
                                                   zoom_range=0.2,  # zoom
                                                   horizontal_flip=True,  # horizontal flip
                                                   brightness_range=[0.2, 0.8])  # brightness

                self.hist = self.model.fit(
                    train_datagen.flow(x_train, y_train,
                                       batch_size=self.opt.batch_size,
                                       seed=27,
                                       shuffle=False),
                    epochs=self.opt.epochs,
                    steps_per_epoch=x_train.shape[0] // self.opt.batch_size,
                )
            else:

                self.hist = self.model.fit(
                    x_train, y_train, epochs=self.opt.epochs, batch_size=self.opt.batch_size)
        else:
            if self.opt.data_augumentation:
                train_datagen = ImageDataGenerator(rotation_range=5,  # rotation
                                                   width_shift_range=0.2,  # horizontal shift
                                                   zoom_range=0.2,  # zoom
                                                   horizontal_flip=True,  # horizontal flip
                                                   brightness_range=[0.2, 0.8])  # brightness
                self.hist = self.model.fit(
                    train_datagen.flow(x_train, y_train,
                                       batch_size=self.opt.batch_size,
                                       seed=27,
                                       shuffle=False),
                    epochs=self.opt.epochs,
                    steps_per_epoch=x_train.shape[0] // self.opt.batch_size,
                    validation_split=0.1
                )
            else:
                self.hist = self.model.fit(
                    x_train, y_train, epochs=self.opt.epochs, batch_size=self.opt.batch_size,  validation_split=0.1)

    def test(self, x_test, y_test):
        _, self.metrics['accuracy'] = self.model.evaluate(x_test, y_test)
        if not self.run:
            self.metrics["acc_val"] = self.hist.history['val_accuracy'][-1]

        self.probs = self.model.predict(x_test)
        self.y_pred = self.probs.argmax(axis=-1)
        self.y_test = y_test.argmax(axis=-1)
        self.pred_bool = self.y_pred == self.y_test
        self.x_test = x_test

    def compile(self, loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
        optimizer_dicc = {
            'sgd': keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
            'rmsprop': keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
            'adagrad': keras.optimizers.Adagrad(learning_rate=0.01, epsilon=1e-08, decay=0.0),
            'adadelta': keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
            'adam': keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)}

        self.optmizer_setup = optimizer
        self.model.compile(loss=loss,
                           optimizer=optimizer_dicc[optimizer], metrics=metrics)

    def save_metrics(self):
        self._compute_metrics()
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y %H:%M:%S")
        metrics = dict(
            dataset=self.dataset_name,
            model=self.opt.model_name,
            date_time=date_time,
            epochs=self.opt.epochs,
            batch_size=self.opt.batch_size,
            training_mode=self.opt.training_mode,
            acc_test=self.metrics['accuracy'],
            acc_train=self.hist.history['accuracy'][-1],
            precision=self.metrics['precision'],
            recall=self.metrics['recall'],
            fscore=self.metrics['fscore'],
            kappa=self.metrics['kappa'],
            auc=self.metrics['auc'],
            auc_precision_recall=self.metrics['auc_precision_recall'],
            optmizer_setup=self.optmizer_setup,
            data_augumentation=self.opt.data_augumentation,
            experiment_code=self.artifact_folder.split("artifacts/")[1],
            experiment_git_hash=self._get_experiment_git_commit_hash(),
            experiment_git_branch=self._get_experiment_branch(),
        )
        if self.subject:
            metrics["subject"] = str(self.subject)
        elif self.run:
            metrics["run"] = self.run
            metrics["folds"] = self.opt.cross_validation_folds
        else:
            metrics["acc_val"] = self.metrics['acc_val']

        print(metrics)
        self._make_saving_dir_if_not_exists()
        with open(self._get_save_directory() + "/metrics.json", "w") as outfile:
            json.dump(metrics, outfile)

    def save_predictions(self):
        self._make_saving_dir_if_not_exists()
        np.save(self._get_save_directory() + '/y_test', self.y_test)
        np.save(self._get_save_directory() + '/y_pred', self.y_pred)
        np.save(self._get_save_directory() + '/y_prob', self.probs)
        np.save(self._get_save_directory() + '/pred_bool', self.pred_bool)

    def save_model(self):
        self._make_saving_dir_if_not_exists()
        self.model.save(self._get_save_directory() + '/model')

    def set_dataset(self, dataset):
        self.dataset_name = dataset

    def set_save_dir(self, save_dir):
        self.artifact_folder = save_dir + self.opt.model_name + '_' + self.dataset_name

    def set_subject(self, subj):
        self.subject = subj

    def set_run(self, run):
        self.run = str(run)

    def save_plots(self):
        draw_confusion_matrix(
            labels=self.y_test,
            y_pred=self.y_pred,
            results_path=self._get_save_directory(),
            display_labels=self.opt.labels
        )
        draw_learning_curves(
            history=self.hist,
            results_path=self._get_save_directory()
        )
        draw_roc_curves(
            metrics=self.metrics,
            results_path=self._get_save_directory()
        )

    def _get_save_directory(self):
        print(f'Saving in -> {self.artifact_folder}')
        if self.opt.training_mode == 'intra-subject':
            return self.artifact_folder + f'/subj_{self.subject}'

        elif self.opt.training_mode == 'cross-validation':
            return self.artifact_folder + f'/run_{self.run}'
        return self.artifact_folder

    def _make_saving_dir_if_not_exists(self):
        new_path = pathlib.Path(self._get_save_directory())
        new_path.mkdir(parents=True, exist_ok=True)

    def _compute_metrics(self):
        self.metrics['precision'], self.metrics['recall'], self.metrics['fscore'], self.metrics['support'] = metrics.precision_recall_fscore_support(
            self.y_test, self.y_pred, average='weighted')
        self.metrics['kappa'] = metrics.cohen_kappa_score(
            self.y_test, self.y_pred)
        self.metrics['fpr'], self.metrics['tpr'], thresholds = metrics.roc_curve(
            self.y_test, self.probs[:, 1], pos_label=1)
        self.metrics['auc'] = metrics.auc(
            self.metrics['fpr'], self.metrics['tpr'])
        self.metrics['precision_C'], self.metrics['recall_C'], thresholds = metrics.precision_recall_curve(
            self.y_test, self.probs[:, 1], pos_label=1)
        # Use AUC function to calculate the area under the curve of precision recall curve
        self.metrics['auc_precision_recall'] = metrics.auc(
            self.metrics['recall_C'], self.metrics['precision_C'])

    def _get_experiment_git_commit_hash(self):
        try:
            process = Popen(["git", "rev-parse", "HEAD"], stdout=PIPE)
            (commit_hash, err) = process.communicate()
            process.wait()
            return str(commit_hash)
        except Exception as e:
            print(e)
            return 'Na'

    def _get_experiment_branch(self):
        from git import Repo
        try:
            repo = Repo(os.getcwd().replace('notebooks', ''))
            branch = repo.active_branch
            return branch.name
        except Exception as e:
            print(e)
            return 'Na'
