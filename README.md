# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Include the problem statement and Dataset
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:

Import the required libraries (PyTorch, torchvision, matplotlib, etc.) and set up the device (CPU/GPU).

### STEP 2:

Load the dataset (train and test). Apply transformations such as resizing, normalization, and augmentation. Create `DataLoader` objects.

### STEP 3:

Load the pre-trained VGG-19 model from `torchvision.models`. Modify the **final fully connected layer** to match the number of classes in the dataset.

### STEP 4:

Define the **loss function** (CrossEntropyLoss) and the **optimizer** (Adam).

### STEP 5:

Train the model for the required number of epochs while recording **training loss** and **validation loss**.

### STEP 6:

Evaluate the model using a **confusion matrix**, **classification report**, and test it on **new samples**.

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

!unzip -qq ./chip_data.zip -d data

dataset_path = "./data/dataset/"

train_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/test",
    transform=transform
)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader  = DataLoader(test_dataset,batch_size=32,shuffle=False)
from torchvision.models.vgg import VGG19_Weights

model = models.vgg19(weights=VGG19_Weights.DEFAULT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Modify the final fully connected layer to match the dataset classes
model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features,
    1
)

# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False


# Include the Loss function and optimizer


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader, test_loader, epochs=10):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # -------- TRAIN --------
        model.train()
        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss = val_loss / len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print("Name:latchaya priyan S       ")
    print("Register Number: 212224230139       ")
    # Plot Loss
    plt.plot(train_losses,label="Train Loss")
    plt.plot(val_losses,label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.show()

def test_model(model,test_loader):

    model.eval()
    all_preds=[]
    all_labels=[]
    correct=0
    total=0

    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            probs=torch.sigmoid(outputs)
            preds=(probs>0.5).int().squeeze()

            correct+=(preds==labels).sum().item()
            total+=labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("Name:latchaya priyan S       ")
    print("Register Number: 212224230139       ")
    print("Test Accuracy:",correct/total)

    cm=confusion_matrix(all_labels,all_preds)

    sns.heatmap(cm,annot=True,fmt="d",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(
        all_labels,all_preds,
        target_names=train_dataset.classes))
def predict_image(model,index,dataset):

    model.eval()
    image,label=dataset[index]

    with torch.no_grad():
        output=model(image.unsqueeze(0).to(device))
        prob=torch.sigmoid(output)
        pred=(prob>0.5).int().item()
    print("Name:latchaya priyan S       ")
    print("Register Number: 212224230139       ")
    plt.imshow(transforms.ToPILImage()(image))
    plt.title(f"Actual:{dataset.classes[label]} | Predicted:{dataset.classes[pred]}")
    plt.axis("off")
    plt.show()

predict_image(model,55,test_dataset)
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="726" height="728" alt="image" src="https://github.com/user-attachments/assets/d61dec0c-f4fb-4669-aacc-9537a2c530e6" />

</br>
</br>
</br>

### Confusion Matrix
<img width="587" height="551" alt="image" src="https://github.com/user-attachments/assets/6e186157-e409-4677-8020-87876f68abf6" />

</br>
</br>
</br>

### Classification Report
<img width="487" height="202" alt="image" src="https://github.com/user-attachments/assets/90bb739c-b66d-43fd-852f-38f02a07a92f" />

</br>
</br>
</br>

### New Sample Prediction
<img width="451" height="842" alt="image" src="https://github.com/user-attachments/assets/7ce196f4-a97e-4a0b-83d1-1ded8364a744" />

</br>
</br>
</br>

## RESULT
Thus, Transfer Learning using VGG-19 was successfully implemented for image classification. The model was fine-tuned on the given dataset, and evaluation using confusion matrix, classification report, and predictions on new samples confirmed its effectiveness. Transfer learning significantly reduced training time and improved accuracy compared to training from scratch.
</br>
</br>
</br>
