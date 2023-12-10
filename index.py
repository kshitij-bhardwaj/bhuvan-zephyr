import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
def load_and_visualize_images(data_folder):
    # List all files in the data folder
    files = os.listdir(data_folder)

    # Iterate through each file in the folder
    for file in files:
        # Check if the file is a TIFF image
        if file.lower().endswith(('.tiff', '.tif')):
            # Construct the file path
            file_path = os.path.join(data_folder, file)

            # Read the TIFF image using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Display the image
            plt.figure(figsize=(8, 8))
            plt.imshow(image, cmap='hsv')
            plt.title(f'TIFF Image: {file}')
            plt.show()




def load_and_preprocess_images(data_folder):
    # List all files in the data folder
    files = os.listdir(data_folder)

    # Initialize lists to store images and labels
    images = []
    labels = []

    # Iterate through each file in the folder
    for file in files:
        # Check if the file is a TIFF image
        if file.lower().endswith(('.tiff', '.tif')):
            # Construct the file path
            file_path = os.path.join(data_folder, file)

            # Read the TIFF image using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Preprocess the image (add additional preprocessing steps as needed)
            # For example, you might resize the image to a specific size
            image = cv2.resize(image, (224, 224))

            # Append the preprocessed image to the list
            images.append(image)

            # Extract label information (if available)
            # You might need to define a mechanism to extract labels based on your dataset structure
            # labels.append(extract_label_from_filename(file))

    # Convert the lists to NumPy arrays
    images = np.array(images)
    # labels = np.array(labels)

    # Split the data into training and testing sets
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
    # train_labels, test_labels = train_test_split(labels, test_size=0.2, random_state=42)

    return train_images, test_images  # Return additional variables as needed


if __name__ == "__main__":
    # Replace 'your_data_folder' with the actual path to your TIFF images folder
    data_folder = 'topic13/LULC_2005_15_vik'
    # Call the function to load and visualize images
    load_and_visualize_images(data_folder)
     # Call the function to load and preprocess images
    train_images, test_images = load_and_preprocess_images(data_folder)
