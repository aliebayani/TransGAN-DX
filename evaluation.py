from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

def evaluate_model(classifier, test_loader):
    classifier.eval()
    true_labels, predictions = [], []

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            outputs = classifier(batch_inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(batch_labels.numpy())
            predictions.extend(predicted.numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, predictions)

    print(f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")

    return true_labels, predictions
