import os
import random
import shutil
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#%% Function to source images
def source_images(orig_dir, dest_dir, num_images=300, seed=23):
    # Set seed for reproducibility
    random.seed(seed)

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Get a list of all files in the source directory
    all_files = os.listdir(orig_dir)
    
    # Select the specified number of images randomly
    selected_files = random.sample(all_files, num_images)
    
    # Copy the selected files to the destination directory
    for file in selected_files:
        shutil.copy(os.path.join(orig_dir, file), os.path.join(dest_dir, file))

#%% Function to visualize training histories
def calc_histories(metrics, histories, history_names):
    for metric in metrics:
        fig_name = f'fig_{metric}'
        fig = plt.figure()
        for history, name in zip(histories, history_names):
            plt.plot(history[metric], label = name)
        plt.title(f'{metric.capitalize()} Across Models')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        fig.show()
        fig.savefig('/kaggle/working/' + fig_name +'.png')

#%% Function to calculate evaluation metrics
def calc_eval_metrics(model, name, test_generator):
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    row = pd.DataFrame({'Model Name': name, 'Test Accuracy': [accuracy], 'Test Precision': [precision], 'Test Recall': [recall], 'Test F1': [f1]})

    return row, accuracy, precision, recall, f1, y_true, y_pred_classes, y_pred

#%% Function to calculate confusion matrix metrics
def calc_confusion_matrix(y_true, y_pred_classes, fig_name):
    cm = confusion_matrix(y_true, y_pred_classes)
    fig = plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(fig_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    output_path = '/kaggle/working/output/model_metrics/'
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_path + fig_name +'.png')

#%% Function to calculate ROC curve
def calc_roc_curve(n_classes, y_true, y_pred, fig_name):
    fpr = dict() #Fale positive
    tpr = dict() #True positive
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig = plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(fig_name)
    plt.legend(loc="lower right")
    plt.show()

    output_path = '/kaggle/working/output/model_metrics/'
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_path + 'ROC_' + fig_name +'.png')