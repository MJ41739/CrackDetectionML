import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import os

class CrackDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crack Detection Application")
        self.dataset = None
        self.model = None
        self.encoder = None

        # GUI Elements
        self.load_data_button = tk.Button(root, text="Load Dataset", command=self.load_dataset)
        self.load_data_button.pack(pady=10)

        self.train_model_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict Crack", command=self.predict_image)
        self.predict_button.pack(pady=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.dataset = pd.read_csv(file_path)
            if 'Image File Path' not in self.dataset.columns or 'Labels' not in self.dataset.columns:
                raise ValueError("Dataset must have 'Image File Path' and 'Labels' columns.")

            messagebox.showinfo("Success", "Dataset loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading dataset: {e}")

    def preprocess_images(self, df):
        images = []
        labels = []

        for _, row in df.iterrows():
            image_path = row['Image File Path']
            label = row['Labels']

            if not os.path.exists(image_path):
                continue

            # Preprocess image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0

            images.append(image)
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def train_model(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        try:
            X, y = self.preprocess_images(self.dataset)

            self.encoder = LabelEncoder()
            y_encoded = self.encoder.fit_transform(y)
            y_categorical = to_categorical(y_encoded)

            X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

            self.model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

            messagebox.showinfo("Success", "Model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {e}")

    def predict_image(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png")])
        if not file_path:
            return

        try:
            image = cv2.imread(file_path)
            image = cv2.resize(image, (128, 128))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = self.model.predict(image)
            label_index = np.argmax(prediction, axis=1)[0]
            label = self.encoder.inverse_transform([label_index])[0]

            messagebox.showinfo("Prediction", f"The image is: {label}")
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CrackDetectionApp(root)
    root.geometry("400x300")
    root.mainloop()
