import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

dataset_path = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/agument/gray_1024_his_aug"  

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

        glcm = graycomatrix(
            img,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )
        contrast      = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity   = graycoprops(glcm, 'homogeneity')[0, 0]
        energy        = graycoprops(glcm, 'energy')[0, 0]
        correlation   = graycoprops(glcm, 'correlation')[0, 0]

        features.append({
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation,
            'class': class_name
        })

output_dir = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/ekstraksi_fitur"
os.makedirs(output_dir, exist_ok=True)  
output_path = os.path.join(output_dir, 'glcm_features2.csv')

df = pd.DataFrame(features)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} feature vectors to {output_path}")

