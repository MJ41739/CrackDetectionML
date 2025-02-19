import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tkinter import filedialog, messagebox
import tkinter as tk
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from PIL import Image

# Custom Dataset for YOLO
class CrackDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        label_encoder = LabelEncoder()
        label_encoder.fit(['cracked', 'not cracked'])

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        bbox = self.img_labels.iloc[idx, 2]  # Assuming bounding box data is in column 2

        # Encode label
        label = label_encoder.transform([label])[0]

        # Convert bounding box string to tensor
        bbox = torch.tensor(eval(bbox), dtype=torch.float32).unsqueeze(0)  # Ensure shape [N, 4]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label)  # Ensure label is a tensor
        return image, {"boxes": bbox, "labels": torch.tensor([label])}

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

# Prepare YOLO Model
def prepare_yolo_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # Background and Crack
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# Train YOLO Model
def train_yolo(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

# Predict Function
def predict_yolo(model, image_path, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image)

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    return boxes, scores, labels

# GUI Integration
class CrackDetectionAppYOLO:
    def __init__(self, root):
        self.root = root
        self.root.title("Crack Detection using YOLO")

        self.dataset_path = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GUI Elements
        self.load_data_button = tk.Button(root, text="Load Dataset", command=self.load_dataset)
        self.load_data_button.pack(pady=10)

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict Crack", command=self.predict_image)
        self.predict_button.pack(pady=10)

    def load_dataset(self):
        self.dataset_path = filedialog.askopenfilename(title="Select Dataset CSV", filetypes=[("CSV Files", "*.csv")])
        if not self.dataset_path:
            messagebox.showerror("Error", "No dataset selected.")
            return
        messagebox.showinfo("Success", "Dataset loaded successfully.")

    def train_model(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Please load the dataset first.")
            return

        try:
            transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
            ])

            dataset = CrackDataset(self.dataset_path, "./images", transform=transform)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

            self.model = prepare_yolo_model().to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            train_yolo(self.model, dataloader, optimizer, self.device)

            messagebox.showinfo("Success", "Model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")

    def predict_image(self):
        if not self.model:
            messagebox.showerror("Error", "Model is not trained yet.")
            return

        image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png")])
        if not image_path:
            return

        try:
            boxes, scores, labels = predict_yolo(self.model, image_path, self.device)
            print("Boxes:", boxes)
            print("Scores:", scores)
            print("Labels:", labels)

            messagebox.showinfo("Prediction Result", f"Prediction completed. Check console for details.")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CrackDetectionAppYOLO(root)
    root.geometry("400x300")
    root.mainloop()
