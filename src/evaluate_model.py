import tensorflow as tf  
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import os
import utils


parent_dir = os.path.dirname(os.path.abspath(__file__))  


def evaluate_model(model, x_test, y_test, label_map, model_type):
    y_pred_probs = model.predict(x_test)  # get probability scores
    y_pred = np.argmax(y_pred_probs, axis=1)  # convert to class labels


    cm = confusion_matrix(y_test, y_pred)
   
    # Calculate sensitivity and specificity
    sensitivity = recall_score(y_test, y_pred, average='weighted' )
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  
    ppv = precision_score(y_test, y_pred, average='weighted')  
    f1 = f1_score(y_test, y_pred, average='weighted')  
    auc_score = roc_auc_score(y_test, y_pred_probs, multi_class='ovr', average='weighted')


    plot_confusion_matrix(cm, sensitivity, specificity, model_type, label_map)
    plot_roc_curve(model, x_test, y_test, model_type, label_map)
    plot_precision_recall_curve(model, x_test, y_test, model_type, label_map)
   
    return y_pred, cm


def plot_confusion_matrix(cm, sensitivity, specificity, model_type, label_map):
    plt.figure(figsize=(8, 6))


    cm_log = np.log1p(cm)  # colour scale, origin counts


    class_labels = sorted(label_map, key=label_map.get)


    sns.heatmap(cm_log, annot=cm, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_labels,  yticklabels=class_labels,
                annot_kws={"size": 10})  # orig counts
    plt.title(f'Confusion Matrix: {model_type} CNN\nSensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(parent_dir, f"confusion_matrix_{model_type}.png"))
    plt.close()


def plot_roc_curve(model, X_test, y_test, model_type, label_map):
    y_test_bin = label_binarize(y_test, classes=[i for i in range(len(np.unique(y_test)))])
   
    y_pred_probs = model.predict(X_test)
   
    fpr, tpr, roc_auc = {}, {}, {}
   
    for i in range(y_test_bin.shape[1]):
      fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
      auc_value = auc(fpr[i], tpr[i])  
      roc_auc[i] = auc_value  
   
    plt.figure(figsize=(8, 6))
    for i in range(y_test_bin.shape[1]):
        class_name = sorted(label_map, key=label_map.get)[i]  
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_name} (AUC = {roc_auc[i]:.2f})')


    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({model_type} CNN)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(parent_dir, f"roc_curve_{model_type}.png"))
    plt.close()


def plot_precision_recall_curve(model, X_test, y_test, model_type, label_map):
    y_test_bin = label_binarize(y_test, classes=[i for i in range(len(np.unique(y_test)))])
   
    y_pred_probs = model.predict(X_test)
   
    plt.figure(figsize=(8, 6))
   
    for i in range(y_test_bin.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
        class_name = sorted(label_map, key=label_map.get)[i]
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AUC = {auc(recall, precision):.2f})')


    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title(f'Precision-Recall Curve ({model_type} CNN)')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(parent_dir, f"precision_recall_curve_{model_type}.png"))
    plt.close()


def main():
    model_1d = os.path.join(parent_dir, "1D_CNN.h5")  
    model_1d = tf.keras.models.load_model(model_1d)


    model_2d = os.path.join(parent_dir, "2D_CNN.h5")  
    model_2d = tf.keras.models.load_model(model_2d)
    print("Successfully loaded models for testing")


    database_path = os.path.join(parent_dir, "ecg_analysis.db")
    _, test_imgs, _, test_series, _, test_labels, label_map = utils.split_data(database_path)
   
    y_pred_2d, cm_2d = evaluate_model(model_2d, test_imgs, test_labels, label_map, "2D")
    y_pred_1d, cm_1d = evaluate_model(model_1d, test_series, test_labels, label_map, "1D")


if __name__ == "__main__":
    main()
