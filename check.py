import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt 

df = pd.read_csv('dataset.csv')
print(df['Labels'].value_counts())


df = pd.read_csv('dataset.csv')
sample_path = df.iloc[0]['Image File Path']
img = cv2.imread(sample_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(df.iloc[0]['Labels'])
plt.show()

from sklearn.preprocessing import LabelEncoder

df['Labels'] = LabelEncoder().fit_transform(df['Labels'])
print(df['Labels'].unique())  # Output should be integers (e.g., 0, 1)


# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Example: Augment a single image
# image = ...  # Load an image as a numpy array
# augmented_image = datagen.random_transform(image)
