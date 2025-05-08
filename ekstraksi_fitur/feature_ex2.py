import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

dataset_path = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/agument/gray_1024_his_aug"  
output_dir = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/ekstraksi_fitur"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'glcm_features_ex.csv')

distances = [1, 2, 3]
angles_deg = [0, 45, 90, 135]
angles_rad = [np.deg2rad(angle) for angle in angles_deg]

features = []

for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue
        img_path = os.path.join(class_dir, fname)

        img = io.imread(img_path)
        if img.ndim == 3:
            img = rgb2gray(img)
            img = (img * 255).astype('uint8')
        else:
            img = img.astype('uint8')

        # Compute GLCM with all distances and angles
        glcm = graycomatrix(
            img,
            distances=distances,
            angles=angles_rad,
            levels=256,
            symmetric=True,
            normed=True
        )

        # Initialize feature entry with class
        feature_entry = {'class': class_name}

        # Calculate properties for each distance and angle
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in properties:
            prop_matrix = graycoprops(glcm, prop)  # Shape (num_distances, num_angles)
            for d_idx, d in enumerate(distances):
                for a_idx, a in enumerate(angles_deg):
                    key = f"{prop}_d{d}_a{a}"
                    feature_entry[key] = prop_matrix[d_idx, a_idx]

        features.append(feature_entry)

# Save to CSV
df = pd.DataFrame(features)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} feature vectors to {output_path}")