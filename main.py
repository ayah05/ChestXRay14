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
from model import ResNet50
from dataset import ChestXRayDataset
from PIL import Image, ImageDraw
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

# load data
path = kagglehub.dataset_download("nih-chest-xrays/data")
__labels = pd.read_csv(os.path.join(path, "Data_Entry_2017.csv"))
print("Shape of Labels at the beginning:", __labels.shape)
images_file_path = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(".png")]
image_filenames = set(os.path.basename(img) for img in images_file_path)

# drop unnecessary/invalid data
__labels = __labels[__labels['Image Index'].isin(image_filenames)]
__labels = __labels.drop(['Unnamed: 11'], axis=1)
labels_filtered = __labels[__labels['Patient Age'] <= 95]
labels_filtered = pd.get_dummies(labels_filtered, columns=['Patient Gender', 'View Position'])

# select 10 labels with most counts
top_10_labels = labels_filtered['Finding Labels'].value_counts().head(10).index.tolist()
__labels = labels_filtered[labels_filtered['Finding Labels'].isin(top_10_labels)]
print("Shape of Labels after selecting Top 10 Counts", __labels.shape)

# plot 10 diseases with most counts
fig, axes = plt.subplots(2, 5, figsize=(15,6))
axes = axes.flatten()
for i, label in enumerate(top_10_labels[:len(axes)]):
    rows_with_label = __labels[__labels['Finding Labels'] == label]
    image_path = None
    for _, row in rows_with_label.iterrows():
        matching_files = [img for img in images_file_path if os.path.basename(img) == row['Image Index']]
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

# plot classes where bbox is available
bbox_list_path = "BBox_List_2017.csv"
bbox_list_csv = os.path.join(path, bbox_list_path)
bbox_df = pd.read_csv(bbox_list_csv)
bbox_df = bbox_df.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], axis=1)
categories = bbox_df['Finding Label'].unique()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i, category in enumerate(categories):
    category_row = bbox_df[bbox_df['Finding Label'] == category].iloc[0]
    image_index = category_row['Image Index']
    x = category_row['Bbox [x']
    y = category_row['y']
    w = category_row['w']
    h = category_row['h]']
    image_path = next((img for img in images_file_path if os.path.basename(img) == image_index), None)
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

# select 400 rows per class randomly
__data = pd.concat([group.sample(n=min(len(group), 400), random_state=42) for _, group in __labels.groupby('Finding Labels')])

# selecting metadata columns
metadata_columns = ['Patient Gender_F', 'Patient Gender_M', 'View Position_AP', 'View Position_PA', 'Patient Age', 'Image Index']

# apply min-max normalization on age
min_age = __data['Patient Age'].min()
max_age = __data['Patient Age'].max()
__data['Patient Age'] = (__data['Patient Age'] - min_age) / (max_age - min_age)

# extracting metadata data
metadata = __data[metadata_columns]

# drop target value and Patient ID because it has around 2800 unique values
X = __data.drop(columns=['Finding Labels','Patient ID', 'Patient Gender_F', 'Patient Gender_M', 'View Position_AP', 'View Position_PA'], axis=1)
y = __data['Finding Labels']

# set 'Image Index' as index for both X and metadata
X = X.set_index('Image Index')
metadata = metadata.set_index('Image Index')
metadata = metadata.astype(float)

# train - test - split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
metadata_train = metadata.loc[X_train.index]
metadata_test = metadata.loc[X_test.index]

# label encode target value
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# transforms
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


# dataset and dataLoader
train_dataset = ChestXRayDataset(X_train, y_train_encoded, images_file_path, metadata_train, train_transform)
test_dataset = ChestXRayDataset(X_test, y_test_encoded, images_file_path, metadata_test, test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# verifying shape of data
for images, labels, metadata in train_dataloader:
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Metadata shape:", metadata.shape)
    break

# initializing device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_channels=1).to(device)

# defining optimizer, criterion and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# defining epoch
num_epochs = 20
best_loss = float('inf')

# train_losses = []
# train_accuracies = []
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     # training loop
#     for images, labels, metadata in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#         images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
#         optimizer.zero_grad()
#         outputs = model(images, metadata)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         preds = (outputs > 0.5).float()
#         correct += (preds == labels).sum().item()
#         total += labels.numel()
#
#     avg_train_loss = running_loss / len(train_dataloader)
#     train_accuracy = correct / total
#
#     # validation loop
#     model.eval()
#     validation_loss = 0.0
#     with torch.no_grad():
#         for images, labels, metadata in test_dataloader:
#             images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
#             outputs = model(images, metadata)
#             loss = criterion(outputs, labels)
#             validation_loss += loss.item()
#
#     avg_validation_loss = validation_loss / len(test_dataloader)
#     scheduler.step(avg_validation_loss)
#
#     print(f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}, Accuracy: {train_accuracy:.4f}")
#
#     # save best model
#     if avg_validation_loss < best_loss:
#         best_loss = avg_validation_loss
#         torch.save(model.state_dict(), "resnet50_.pth")
#         print("\nModel saved!")

model.load_state_dict(torch.load('resnet50_.pth'))
model.eval()

test_loss = 0.0
correct = 0
total = 0

# testing
true_positives = torch.zeros(10).to(device)
false_positives = torch.zeros(10).to(device)
false_negatives = torch.zeros(10).to(device)

all_labels = []
all_probs = []
all_preds = []
with torch.no_grad():
    for images, labels, metadata in tqdm(test_dataloader, desc="Testing"):
        images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)

        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        probs = torch.sigmoid(outputs)
        print("Model Logits:", outputs[:5])
        print("Sigmoid Outputs:", torch.sigmoid(outputs)[:5])
        test_loss += loss.item()
        preds = (probs > 0.5).float()
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())

        print("Batch Predictions:", preds[:5])
        print("Batch Labels:", labels[:5])

        true_positives += (preds * labels).sum(dim=0)
        false_positives += (preds * (1 - labels)).sum(dim=0)
        false_negatives += ((1 - preds) * labels).sum(dim=0)

        correct += (preds == labels).sum().item()
        total += labels.numel()

print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)

test_loss /= len(test_dataloader)
precision = true_positives / (true_positives + false_positives + 1e-7)
recall = true_positives / (true_positives + false_negatives + 1e-7)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
accuracy = correct / total

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision.mean().item():.4f}, Recall: {recall.mean().item():.4f}, F1 Score: {f1_score.mean().item():.4f}")


all_labels = torch.cat(all_labels, dim=0).numpy()
all_probs = torch.cat(all_probs, dim=0).numpy()
all_preds = torch.cat(all_preds, dim=0).numpy()

# plot roc curve
n_classes = 10
label_names = label_encoder.classes_
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
    auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
    plt.plot(fpr, tpr, label=f"Class {label_names[i]} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Each Label")
plt.legend(loc="best")
plt.grid()
plt.show()


# plot confusion matrices
confusion_matrices = multilabel_confusion_matrix(all_labels, all_preds)
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i, (cm, label) in enumerate(zip(confusion_matrices, label_names)):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Not {label}", label])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f"Confusion Matrix for {label}", fontsize=12)
    axes[i].set_xlabel("Predicted Label", fontsize=10)
    axes[i].set_ylabel("True Label", fontsize=10)
    axes[i].tick_params(axis='both', labelsize=8)

for j in range(len(label_names), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3.0)
plt.show()