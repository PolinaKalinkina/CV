
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import pandas as pd
import os

yolo_model = YOLO('yolov8n.pt')

cnn_model = models.resnet50(pretrained=True)
cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])  # Убираем последний слой
cnn_model.eval()

def extract_features(image):
    """Извлечение признаков из изображения с использованием ResNet50."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = cnn_model(image)
    return features.numpy()

def compare_logos(features1, features2):
    """Сравнение признаков двух логотипов с использованием косинусного сходства."""
    similarity = cosine_similarity(features1, features2)
    return similarity[0][0]

def detect_and_compare(image_path, sample_logo_path, similarity_threshold=0.9):
    """Детекция логотипов на изображении и сравнение их с образцом."""

    image = cv2.imread(image_path)


    results = yolo_model(image_path)
    detected_logos = results.boxes


    detected_features = []
    for box in detected_logos:
        x_min, y_min, x_max, y_max = box.xyxy[0]
        logo = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        detected_features.append(extract_features(logo))


    sample_image = cv2.imread(sample_logo_path)
    sample_features = extract_features(sample_image)


    similarities = [compare_logos(features, sample_features) for features in detected_features]


    for i, similarity in enumerate(similarities):
        if similarity > similarity_threshold:  # Порог сходства
            print(f"Логотип {i+1}: Да (сходство: {similarity})")
        else:
            print(f"Логотип {i+1}: Нет (сходство: {similarity})")

def fine_tune_model(new_data, new_labels, epochs=10, batch_size=32):
    """Дообучение модели на новых данных  PyTorch."""

    base_model = models.resnet50(pretrained=True)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, 1)  # Изменяем последний слой для бинарной классификации


    for param in base_model.parameters():
        param.requires_grad = False


    for param in base_model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(base_model.fc.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()


    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    dataset = torch.utils.data.TensorDataset(torch.stack([preprocess(image) for image in new_data]), torch.tensor(new_labels, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    base_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")

    return base_model

class LogoDataset(Dataset):
    """Класс для загрузки и преобразования датасета LogoDet-3K."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = cv2.imread(img_name)
        label = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
from IPython.display import display

def load_dataset(csv_file, root_dir):
    """Загрузка датасета LogoDet-3K."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = LogoDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    return dataset
file_path= lyly99_logodet3k_path

dataset_dir = "/kaggle/input/logodet3k/LogoDet-3K"
df = pd.DataFrame(glob.glob(f"{dataset_dir}/*/*/*"), columns=["file_path"])
df["ext"] = df["file_path"].apply(lambda x: x.split(".")[-1])
df["logo_category"] = df["file_path"].apply(lambda x: x.split(os.sep)[-3])
df["logo_name"] = df["file_path"].apply(lambda x: x.split(os.sep)[-2])
df.head()

df.to_csv("logodet3k_reference.csv", index=False)
df["ext"].value_counts()
df2 = df[df["ext"]=="jpg"].iloc[:]
print({
    "No. of categories": df2["logo_category"].nunique(),
    "No. of logo types": df2["logo_name"].nunique(),
    "Avg. no. of images per category": df2.groupby("logo_category")["file_path"].count().mean(),
    "Avg. no. of images per logo": df2.groupby("logo_name")["file_path"].count().mean(),
})

df2["logo_category"].value_counts()
df2["logo_name"].value_counts().reset_index().plot(
    x="logo_name", y="count", figsize=(10,5), title="Distribution of logo img counts")

dataset_dst_dir = "/kaggle/working/logodet3k"
if os.path.exists(dataset_dst_dir):
    shutil.rmtree(dataset_dst_dir)
os.makedirs(f"{dataset_dst_dir}/train", exist_ok=True)
os.makedirs(f"{dataset_dst_dir}/val", exist_ok=True)


classname2idx = {logo_name: idx for idx, logo_name in enumerate(sorted(df2["logo_name"].unique()))}
idx2classname = {idx: logo_name for logo_name, idx in classname2idx.items()}

class_name_idx_map_str = "\n".join([f"    {idx}: {class_name}" for class_name, idx in classname2idx.items()])
dataset_config = f"""
path: {dataset_dst_dir}
train:
    - train
val:
    - val

# Classes
names:
{class_name_idx_map_str}
"""

with open("dataset_config.yaml", "w") as f:
    f.write(dataset_config)

def convert_voc_to_yolo(src, dst, classname2idx):
    tree = ET.parse(src)
    root = tree.getroot()
    yolo_lines = []
    image_width = float(root.find("size/width").text)
    image_height = float(root.find("size/height").text)
    depth = float(root.find("size/depth").text)
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        class_index = classname2idx.get(class_name, 0)
        yolo_line = f"{class_index} {x_center} {y_center} {width} {height}"
        yolo_lines.append(yolo_line)
    if dst is not None:
        with open(dst, "w") as f:
            f.write("\n".join(yolo_lines))
    return yolo_lines

df2['is_train'] = True
train_df, test_df = train_test_split(df2, test_size=0.2, random_state=101)
test_df['is_train'] = False
final_df = pd.concat([train_df, test_df])
final_df.reset_index(drop=True, inplace=True)

def copy_to_working(x):
    train_folder = "train" if x["is_train"] else "val"
    src = x["file_path"]
    dst = os.path.join(dataset_dst_dir, train_folder, "__".join(x["file_path"].split(os.sep)[-3:]))
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    if not os.path.exists(dst.replace(".jpg", ".txt")):
        convert_voc_to_yolo(src.replace(".jpg", ".xml"), dst.replace(".jpg", ".txt"), classname2idx)
    return True

with ThreadPoolExecutor() as e:
    for _, row in tqdm(final_df.iterrows()):
        status = e.submit(copy_to_working, dict(row))
        copy_to_working_results.append(status)

copy_to_working_results = final_df.apply(lambda x: copy_to_working(x), axis=1)
copy_to_working_results.sum(), final_df.shape[0]


if __name__ == "__main__":

    csv_file = '/content/logodet3k_reference.csv'
    root_dir = '/content/logodet3k/images'
    dataset = load_dataset(csv_file, root_dir)


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    detect_and_compare('/content/logodet3k/images/sample_image.jpg', '/content/logodet3k/images/sample_logo.jpg')