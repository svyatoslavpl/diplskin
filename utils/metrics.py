from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, roc_auc_score
import numpy as np


def calculate_metrics(y_true, y_pred, y_prob, targetnames):
    report = classification_report(y_true, y_pred, target_names=targetnames)
    print("\nClassification Report:")
    print(report)

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))

    roc_auc_micro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='micro')
    roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    print("Micro ROC AUC score: " + str(roc_auc_micro))
    print("Macro ROC AUC score: " + str(roc_auc_macro))

    for i in range(len(targetnames)):
        r = roc_auc_score(y_true[:, i], y_prob[:, i])
        print("The ROC AUC score of " + targetnames[i] + " is: " + str(r))
