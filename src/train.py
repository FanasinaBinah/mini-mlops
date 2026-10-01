import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(
    root="data/raw", train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root="data/raw", train=False, transform=transform, download=True
)

batch_size = 64
learning_rate =0.0001
epochs = 3

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Device:", device)

model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

mlflow.set_experiment("Fashion-MNIST")
with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    })

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        accuracy = correct / total

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
        mlflow.log_metric("train_loss", average_loss, step=epoch)
        mlflow.log_metric("test_accuracy", accuracy, step=epoch)

    example = next(iter(test_loader))[0][:1].cpu()
    model.to("cpu")
    mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=example,
        serialization_format="pickle",
    )