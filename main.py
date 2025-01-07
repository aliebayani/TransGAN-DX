from data_preprocessing import load_and_preprocess_data
from gan_model import train_gan
from transformer_classifier import train_classifier
from evaluation import evaluate_model
from visualization import plot_heatmap, plot_shap

file_path = "File Path"
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

plot_heatmap(X_train)

# Train GAN
train_gan(X_train, input_dim=X_train.shape[1], hidden_dim=128, batch_size=64, epochs=1000)

# Train Classifier
classifier = train_classifier(X_train, y_train, X_test, y_test, batch_size=32, input_dim=X_train.shape[1], hidden_dim=128, num_classes=2, epochs=200)

# Evaluate
test_loader = None  # Replace with actual DataLoader logic
evaluate_model(classifier, test_loader)

# SHAP
plot_shap(classifier, X_test)
