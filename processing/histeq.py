import os
import cv2

input_folder = r'C:\Users\ASUS VivoBook\OneDrive\Desktop\project_milk\Milk-GLCM\processing\gray_1024'
output_folder = r'C:\Users\ASUS VivoBook\OneDrive\Desktop\project_milk\Milk-GLCM\processing\gray_1024_his'

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith('.jpg'):
            input_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            output_path = os.path.join(output_subfolder, file)
            
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            img_clahe = clahe.apply(img)
            
            cv2.imwrite(output_path, img_clahe)
            print(f'Processed: {input_path} -> {output_path}')

print("Contrast enhancement complete!")