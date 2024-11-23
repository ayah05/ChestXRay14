import kagglehub
import pandas as pd
import numpy as np
import  os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import model
import dataset
from sklearn.preprocessing import LabelEncoder




path = kagglehub.dataset_download("nih-chest-xrays/data")

files = os.listdir(path)
print("Files and directories in '", path, "':")
for file in files:
    print(file)

subdir = "images_001/images"
subdir_path = os.path.join(path, subdir)

files_subdir = os.listdir(subdir_path)
print("Files and directories in '", subdir_path, "':")
for file in files_subdir[:10]:
    print(file)

subdir_label = "Data_Entry_2017.csv"
labels_csv = os.path.join(path, subdir_label)
labels_df = pd.read_csv(labels_csv)
labels_df.head()

image_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png"):
            image_files.append(os.path.join(root, file))

image_filenames = [os.path.basename(img) for img in image_files]
labels_df_filtered = labels_df[labels_df['Image Index'].isin(image_filenames)]
labels_df_sorted = labels_df_filtered.sort_values(by="Image Index").reset_index(drop=True)
labels_df_sorted = labels_df_sorted.drop('Unnamed: 11', axis=1)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, ax in enumerate(axes.flat):
    image_path = image_files[i]
    image = Image.open(image_path).convert("RGB")

    label = labels_df_sorted.iloc[i]["Finding Labels"]

    ax.imshow(image)
    ax.set_title(f"Label: {label}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()

label_counts = labels_df_sorted['Finding Labels'].value_counts()
top_10_labels = label_counts.head(10)
plt.figure(figsize=(10, 6))
top_10_labels.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Top 10 Most Common Labels", fontsize=16)
plt.xlabel("Finding Labels", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(15,6))
axes = axes.flatten()

for i, (label, _) in enumerate(top_10_labels.items()):
    rows_with_label = labels_df_filtered[labels_df_filtered['Finding Labels'] == label]

    image_path = None
    for _, row in rows_with_label.iterrows():
        matching_files = [img for img in image_files if os.path.basename(img) == row['Image Index']]
        if matching_files:
            image_path = matching_files[0]
            break

    if image_path is None:
        print(f"No valid image found for label: {label}")
        continue

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
bbox_df.head()

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


transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15)
])

top_10_labels = top_10_labels.index.tolist()
bbox_unique_labels = bbox_df['Finding Label'].unique().tolist()
final_classes = list(set(top_10_labels + bbox_unique_labels))
print(f"Final classes for the model ({len(final_classes)} total): {final_classes}")

labels_filtered = labels_df_sorted[labels_df_sorted['Finding Labels'].isin(final_classes)]

ages = labels_filtered['Patient Age'].unique()
sorted_ages = np.sort(ages)
labels_filtered = labels_filtered[labels_filtered['Patient Age'] <= 95]
labels_filtered = pd.get_dummies(labels_filtered, columns=['Patient Gender', 'View Position'])

X = labels_filtered.drop(columns=['Finding Labels'], axis=1)
y = labels_filtered['Finding Labels']

X_train_labels, X_test_labels, y_train_labels, y_test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

labelEncoder = LabelEncoder()
y_train_labels_encoded = labelEncoder.fit_transform(y_train_labels)
y_test_labels_encoded = labelEncoder.transform(y_test_labels)

dataset_labels = dataset.ChestXRayDataset(dataframe=labels_filtered,labels=y_train_labels_encoded, image_files=image_files, transform=transform)
dataloader_labels = DataLoader(dataset_labels, batch_size=32, shuffle=True)

train_dataset = dataset.ChestXRayDataset(
    dataframe=X_train_labels,
    labels=y_train_labels_encoded,
    image_files=image_files,
    transform=transform
)

val_dataset = dataset.ChestXRayDataset(
    dataframe=X_test_labels,
    labels=y_test_labels_encoded,
    image_files=image_files,
    transform=transform
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

classes_labels = labels_filtered['Finding Labels'].unique()
num_classes = len(classes_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.ResNet50(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 7
patience = 2

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    train_bar = tqdm(train_dataloader, desc="Training", leave=False)

    for batch_idx, (images, labels) in enumerate(train_bar):
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    val_bar = tqdm(val_dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch_idx, (val_images, val_labels) in enumerate(val_bar):
            val_images, val_labels = val_images.to(device), val_labels.to(device).long()
            val_outputs = model(val_images)

            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()

            _, predicted = torch.max(val_outputs, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted == val_labels).sum().item()

            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = val_loss / len(val_dataloader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Save the best model
    if val_loss < best_loss:
        print(f"Validation loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving model...")
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss. Patience count: {epochs_no_improve}/{patience}")

    # Early stopping
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

print("Loading the best model for evaluation...")
model.load_state_dict(torch.load('best_model.pth'))

# Plot training and validation metrics
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
