import kagglehub
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import model
import dataset
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")

path = kagglehub.dataset_download("nih-chest-xrays/data")
labels_df = pd.read_csv(os.path.join(path, "Data_Entry_2017.csv"))
image_files = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(".png")]
image_filenames = set(os.path.basename(img) for img in image_files)
labels_df = labels_df[labels_df['Image Index'].isin(image_filenames)]
labels_df = labels_df.drop('Unnamed: 11', axis=1)
labels_filtered = labels_df[labels_df['Patient Age'] <= 95]
labels_filtered = pd.get_dummies(labels_filtered, columns=['Patient Gender', 'View Position'])



top_10_labels = labels_filtered['Finding Labels'].value_counts().head(10).index.tolist()
labels_df = labels_filtered[labels_filtered['Finding Labels'].isin(top_10_labels)]


fig, axes = plt.subplots(2, 5, figsize=(15,6))
axes = axes.flatten()
for i, label in enumerate(top_10_labels[:len(axes)]):
    rows_with_label = labels_df[labels_df['Finding Labels'] == label]
    image_path = None
    for _, row in rows_with_label.iterrows():
        matching_files = [img for img in image_files if os.path.basename(img) == row['Image Index']]
        if matching_files:
            image_path = matching_files[0]
            break
    try:
        image = Image.open(image_path).convert("RGB")
        axes[i].imshow(image)
        axes[i].set_title(f"{label}", fontsize=10)
        axes[i].axis("off")
    except Exception as e:
        print(f"Error loading image: {e}")

for j in range(len(top_10_labels), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

bbox_list_path = "BBox_List_2017.csv"
bbox_list_csv = os.path.join(path, bbox_list_path)
bbox_df = pd.read_csv(bbox_list_csv)
bbox_df = bbox_df.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], axis=1)
categories = bbox_df['Finding Label'].unique()

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()
for i, category in enumerate(categories):
    category_row = bbox_df[bbox_df['Finding Label'] == category].iloc[0]
    image_index = category_row['Image Index']
    x = category_row['Bbox [x']
    y = category_row['y']
    w = category_row['w']
    h = category_row['h]']
    image_path = next((img for img in image_files if os.path.basename(img) == image_index), None)
    if not image_path:
        print(f"No image found for category {category}")
        continue
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=5)

    axes[i].imshow(image)
    axes[i].set_title(f"{category}", fontsize=10)
    axes[i].axis("off")

for j in range(len(categories), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


sampled_data = pd.concat([
    group.sample(n=min(len(group), 400), random_state=42)
    for _, group in labels_df.groupby('Finding Labels')
])

metadata_columns = ['Patient Gender_F', 'Patient Gender_M', 'View Position_AP', 'View Position_PA', 'Patient Age']
min_age = sampled_data['Patient Age'].min()
max_age = sampled_data['Patient Age'].max()
sampled_data['Patient Age'] = (sampled_data['Patient Age'] - min_age) / (max_age - min_age)
metadata = sampled_data[metadata_columns]

# Exclude metadata from features
X = sampled_data.drop(columns=['Finding Labels']  + metadata_columns+ ['OriginalImage[Width', 'Height]',
       'OriginalImagePixelSpacing[x', 'y]'], axis=1)
print(X.columns)
print(len(sampled_data['Patient ID'].unique()))
y = sampled_data['Finding Labels']
X = X.reset_index(drop=True)
metadata = metadata.reset_index(drop=True)
# Convert boolean columns to float
metadata = metadata.astype(float)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
metadata_train = metadata.iloc[X_train.index]
metadata_test = metadata.iloc[X_test.index]
print(metadata.dtypes)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Transform
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(448, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset and DataLoader
train_dataset = dataset.ChestXRayDatasetWithMetadata(X_train, y_train_encoded, image_files, metadata_train, train_transform)
test_dataset = dataset.ChestXRayDatasetWithMetadata(X_test, y_test_encoded, image_files, metadata_test, test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.ResNet50Meta(num_classes=10, metadata_features=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
#criterion = nn.CrossEntropyLoss()
# Calculate positive weights for each class
class_counts = y_train_encoded.sum(axis=0)  # Count of each class
total_samples = len(y_train_encoded)
pos_weights = (total_samples - class_counts) / class_counts

# Define the loss
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights).to(device))


# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs = 20
best_loss = float('inf')


train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, metadata in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_dataloader)
    accuracy = correct / total
    scheduler.step()  # Update learning rate

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("\nModel saved!")

model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels, metadata in tqdm(test_dataloader, desc="Testing"):
        images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
        outputs = model(images, metadata)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        preds = outputs.argmax(dim=1)

        # Calculate per-sample correctness
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_loss /= len(test_dataloader)
accuracy = correct / total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
