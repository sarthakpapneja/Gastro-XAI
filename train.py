import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from models.classifier import GastroClassifier
from data.loader import get_data_loaders

def train_model(data_dir, num_epochs=25, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Adjust batch size for CPU to avoid excessive memory usage/latency per step
    if device.type == 'cpu':
        print("Running on CPU. Reducing batch size to 8 and epochs to 5 for demonstration purposes.")
        batch_size = 8
        num_epochs = 5

    # Data Loaders
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}
    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # Model
    model = GastroClassifier(num_classes=len(class_names))
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimize only the fc layer since backbone is frozen
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if isinstance(outputs, torch.Tensor):
                        # Standard model output
                        logits = outputs
                    else:
                        # In case model returns something else (e.g. inception)
                        logits = outputs.logits
                        
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'models/best_model.pth')
                print(f"New best model saved with acc: {best_acc:.4f}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    data_dir = "data/kvasir-dataset-v2" # Adjust if needed
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please run utils/download_data.py first.")
    else:
        train_model(data_dir)
