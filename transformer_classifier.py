import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_classifier(X_train, y_train, X_test, y_test, batch_size, input_dim, hidden_dim, num_classes, epochs):
    train_dataset = TensorDataset(torch.tensor(X_train.values.astype(np.float32)), torch.tensor(y_train.values.astype(np.int64)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    classifier = TransformerClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = classifier(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss / len(train_loader):.4f}")

    return classifier
