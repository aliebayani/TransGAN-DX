import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import shap

def plot_heatmap(X_train):
    correlation_matrix = X_train.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_shap(classifier, X_test_tensor):
    explainer = shap.Explainer(classifier, X_test_tensor)
    shap_values = explainer(X_test_tensor)
    shap.summary_plot(shap_values, X_test_tensor)
