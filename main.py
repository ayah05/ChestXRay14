import kagglehub
import pandas as pd
import numpy as np
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

path = kagglehub.dataset_download("nih-chest-xrays/data")

labels_df = pd.read_csv(os.path.join(path, "Data_Entry_2017.csv"))
image_files = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(".png")]
image_filenames = set(os.path.basename(img) for img in image_files)
labels_df = labels_df[labels_df['Image Index'].isin(image_filenames)]

labels_df = labels_df.drop('Unnamed: 11', axis=1)
labels_filtered = labels_df[labels_df['Patient Age'] <= 95]
labels_filtered = pd.get_dummies(labels_filtered, columns=['Patient Gender', 'View Position'])

top_10_labels = labels_df['Finding Labels'].value_counts().head(10).index.tolist()
labels_df = labels_df[labels_df['Finding Labels'].isin(top_10_labels)]


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
X = sampled_data.drop(columns=['Finding Labels'])
y = sampled_data['Finding Labels']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = dataset.ChestXRayDataset(X_train, y_train_encoded, image_files, transform)
test_dataset = dataset.ChestXRayDataset(X_test, y_test_encoded, image_files, transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.CheXNetFPNGrayscale(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

num_epochs = 20
best_loss = float('inf')


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")


model.load_state_dict(torch.load("best_model.pth"))
print("Training complete. Best model loaded.")

# Testing loop
model.eval()
test_loss = 0.0
correct = 0
total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in tqdm(test_dataloader, desc="Testing"):
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        preds = (outputs > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())

test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss:.4f}")