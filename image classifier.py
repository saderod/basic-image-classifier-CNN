import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# Define the model architecture using functional components
def create_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(64 * 62 * 62, 128),  # Adjust this based on your image size
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Linear(128, 2),  # 2 output classes: tree and not a tree
        nn.LogSoftmax(dim=1)
    )


# Data transformations
def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


# Load dataset
def load_data(data_dir, batch_size=32):
    dataset = datasets.ImageFolder(root=data_dir, transform=get_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Training function
def train(model, device, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')


# Evaluation function
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_dir = 'Documents/coding/basic image classification/tree train'
test_dir = 'Documents/coding/basic image classification/tree test'
train_loader = load_data(train_dir)
test_loader = load_data(test_dir)

# Create model, loss function, and optimizer
model = create_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
print("Starting training...")
train(model, device, train_loader, optimizer, criterion)

# Evaluate the model
print("Evaluating the model...")
evaluate(model, device, test_loader)

print("Training and evaluation complete!")