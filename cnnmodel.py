import os
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import filedialog, messagebox

# Data Preprocessing
def preprocess_images(df):
    images = []
    labels = []

    for _, row in df.iterrows():
        image_path = row['Image File Path']
        label = row['Labels']

        if not os.path.exists(image_path):
            continue

        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0

        images.append(image)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Load Dataset
def load_dataset(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Image File Path' not in df.columns or 'Labels' not in df.columns:
            raise ValueError("Dataset must have 'Image File Path' and 'Labels' columns.")
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")

# Build Model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
def train_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_val, y_val),
        batch_size=32
    )
    return history

def evaluate_model(model, X_test, y_test, history):
    # ---------- Predictions ----------
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # ---------- Accuracy ----------
    accuracy = np.mean(y_pred_labels == y_test_labels)
    print(f"Model Accuracy: {accuracy*100:.2f}%")

    # ---------- F1 Score ----------
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    plt.figure(figsize=(6,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=['Non-Corroded','Corroded'],
        yticklabels=['Non-Corroded','Corroded']
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()

    # ---------- Accuracy Bar Graph ----------
    plt.figure(figsize=(4,4))
    plt.bar(["Accuracy"], [accuracy])
    plt.ylim(0,1)
    plt.title("Model Accuracy")
    plt.show()

    # ---------- F1 Score Bar Graph ----------
    plt.figure(figsize=(4,4))
    plt.bar(["F1 Score"], [f1])
    plt.ylim(0,1)
    plt.title("F1 Score Graph")
    plt.show()

    # ---------- Epoch Accuracy ----------
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure(figsize=(6,4))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------- Epoch Loss ----------
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(6,4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# # Evaluate Model
# def evaluate_model(self):
#     if self.model is None:
#         messagebox.showwarning("Warning", "Please train the model first.")
#         return

#     X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_and_split()
#     if X_test is None:
#         return

#     try:
#         # Make predictions on the test set
#         y_pred = self.model.predict(X_test)
#         y_pred_labels = np.argmax(y_pred, axis=1)
#         y_test_labels = np.argmax(y_test, axis=1)

#         # Counting corroded and non-corroded images
#         corroded_count = sum(y_test_labels == 1)  # Assuming 1 is for corroded
#         non_corroded_count = sum(y_test_labels == 0)  # Assuming 0 is for non-corroded

#         total_images = len(y_test_labels)

#         # Displaying the analysis
#         print(f"Total images tested: {total_images}")
#         print(f"Corroded images: {corroded_count}")
#         print(f"Non-corroded images: {non_corroded_count}")

#         # Create a confusion matrix
#         cm = confusion_matrix(y_test_labels, y_pred_labels)

#         # Plotting confusion matrix as heatmap
#         plt.figure(figsize=(6, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Corroded', 'Corroded'], yticklabels=['Non-Corroded', 'Corroded'])
#         plt.title('Confusion Matrix')
#         plt.ylabel('True Labels')
#         plt.xlabel('Predicted Labels')
#         plt.show()

#         # Classification Report with accuracy
#         report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
#         accuracy = report["accuracy"]
#         print(f"Model Accuracy: {accuracy * 100:.2f}%")

#         # Plotting the accuracy
#         plt.figure(figsize=(4, 4))
#         plt.bar(['Accuracy'], [accuracy], color='green')
#         plt.ylim(0, 1)
#         plt.title("Model Accuracy")
#         plt.ylabel("Accuracy")
#         plt.show()

#         # Show success message
#         messagebox.showinfo("Evaluation", "Model evaluation completed. Check the graphical output.")
    
#     except Exception as e:
#         messagebox.showerror("Error", f"Error during model evaluation: {e}")



# Predict on Single Image
def predict_image(model, encoder, image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    label_index = np.argmax(prediction, axis=1)[0]
    label = encoder.inverse_transform([label_index])[0]

    print(f"Prediction: {label}")
    messagebox.showinfo("Prediction Result", f"The image is: {label}")

# GUI Integration
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

        # Add the new "Evaluate Model" button here
        self.evaluate_model_button = tk.Button(root, text="Evaluate Model", command=lambda: self.evaluate_model())
        self.evaluate_model_button.pack(pady=10)


    def load_dataset(self):
        file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.dataset = load_dataset(file_path)
            messagebox.showinfo("Success", "Dataset loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading dataset: {e}")

    def preprocess_and_split(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return None, None, None, None, None, None

        try:
            X, y = preprocess_images(self.dataset)

            self.encoder = LabelEncoder()
            y_encoded = self.encoder.fit_transform(y)
            y_categorical = to_categorical(y_encoded)

            X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            messagebox.showerror("Error", f"Error preprocessing dataset: {e}")
            return None, None, None, None, None, None

    def train_model(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_and_split()
        if X_train is None:
            return

        try:
            self.model = build_model()
            self.history = train_model(self.model, X_train, y_train, X_val, y_val)
            messagebox.showinfo("Success", "Model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {e}")

    def predict_image(self):
        if self.model is None or self.encoder is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png")])
        if not file_path:
            return

        try:
            predict_image(self.model, self.encoder, file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {e}")
    
    def evaluate_model(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Train model first.")
            return

        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_and_split()

        evaluate_model(self.model, X_test, y_test, self.history)

        messagebox.showinfo("Done", "Model evaluation completed!")

if __name__ == "__main__":
    root = tk.Tk()  # Create the main window here
    app = CrackDetectionApp(root)  # Pass the root window to the app class
    root.geometry("400x300")  # Set the window size
    root.mainloop()  # Start the Tkinter event loop
