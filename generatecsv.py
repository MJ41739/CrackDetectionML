import os
import random
import pandas as pd

# Generate Data for CNN

def generate_csv(image_dir, output_csv):
    data = []
    for label in ['cracked', 'not_cracked']:
        folder = os.path.join(image_dir, label)
        if not os.path.exists(folder):
            print(f"Warning: Directory {folder} does not exist. Skipping.")
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            data.append([file_path, label])
    
    if data:
        df = pd.DataFrame(data, columns=['Image File Path', 'Labels'])
        df.to_csv(output_csv, index=False)
        print(f"CSV saved to {output_csv}")
    else:
        print("No data to save. Check your directory structure.")


# Generate Data for YOLO
# import os
# import pandas as pd
# import random

# def generate_csv(image_dir, output_csv):
#     data = []
#     for label in ['cracked', 'not cracked']:
#         folder = os.path.join(image_dir, label)
#         if not os.path.exists(folder):
#             print(f"Warning: Directory {folder} does not exist. Skipping.")
#             continue
#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)
            
#             # Simulate bounding box data
#             # Replace this logic with actual bounding box data if available
#             bbox = [
#                 random.randint(0, 50),  # x_min
#                 random.randint(0, 50),  # y_min
#                 random.randint(51, 100),  # x_max
#                 random.randint(51, 100)  # y_max
#             ]
#             data.append([file_path, label, bbox])
    
#     if data:
#         df = pd.DataFrame(data, columns=['Image File Path', 'Labels', 'Bounding Box'])
#         df.to_csv(output_csv, index=False)
#         print(f"CSV saved to {output_csv}")
#     else:
#         print("No data to save. Check your directory structure.")

# Usage
generate_csv('C:/Users/Mayur Jadhav/OneDrive/Desktop/CrackDetection/data', 'dataset.csv')
