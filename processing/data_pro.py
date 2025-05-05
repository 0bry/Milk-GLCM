import os
from PIL import Image, ImageOps
import shutil

def process_images(input_dir, output_dir, size=(1024, 1024)):
    # Copy directory structure without files
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir, ignore=shutil.ignore_patterns("*.*"))

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                try:
                    # Open and convert to grayscale
                    image = Image.open(input_path).convert("L")
                    # Fit image with padding to maintain aspect ratio
                    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                    
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    image.save(output_path)
                    print(f"Processed: {relative_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")


input_dataset = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/processing/data"
output_dataset = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/processing/gray_1024"
process_images(input_dataset, output_dataset)
