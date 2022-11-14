from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np



def draw_confusion_matrix(labels, y_pred, results_path, display_labels=None):
    # Generate confusion matrix plot
    
    cf_matrix= confusion_matrix(labels, y_pred)
    # cf_matrix= confusion_matrix(rounded_labels, y_pred, normalize='pred')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                                display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    # plt.title('Confusion Matrix of Subject: ' + sub )
    plt.savefig(results_path + '/cf_matrix.png')
    # plt.show()

def draw_learning_curves(history,results_path):
    legend = ['Train']
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history.keys():
        plt.plot(history.history['val_accuracy'])
        legend.append('val')

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.legend(legend, loc='upper left')
    plt.xlabel('Epoch')
    plt.savefig(results_path + '/learning_curves_acc.png')
    # plt.show()
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(history.history['loss'])
    if 'val_accuracy' in history.history.keys():
        plt.plot(history.history['val_loss'])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(results_path + '/learning_curves_loss.png')
    # plt.show()
    # plt.close()

def draw_roc_curves(metrics,results_path):

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2,label='ROC curve (area = %0.2f)' % metrics['auc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(results_path + '/auc_roc.png')
    # plt.show()
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(metrics['recall_C'], metrics['precision_C'], color='darkorange', lw=2,label='ROC curve (area = %0.2f)' % metrics['auc_precision_recall'])
    plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.title('Precision Recall Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(results_path + '/prec_roc.png')
    # plt.show()

def draw_performance_barchart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])   