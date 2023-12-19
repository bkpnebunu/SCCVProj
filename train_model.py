import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class LocDBDataset(Dataset):
    def __init__(self, labels_file, root_dir, transform=None):
        """
        Args:
            labels_file (string): Path to the labels file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labels = {}
        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                self.labels[parts[0]] = int(parts[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, list(self.labels.keys())[idx])
        image = Image.open(img_name)
        label = self.labels[list(self.labels.keys())[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Replace with your actual file paths
labels_file = 'labels.txt'
root_dir = './locDB'

# Initialize the dataset
dataset = LocDBDataset(labels_file=labels_file, root_dir=root_dir, transform=transform)

# DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load a pre-trained ResNet-101 model and modify it for your number of classes (8 in your case)
model = models.resnet101(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 8)  # 8 classes

import torch.optim as optim
from torch.optim import lr_scheduler

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        # Each epoch has a training phase
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in data_loader:
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)

        print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

# Train the model
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
