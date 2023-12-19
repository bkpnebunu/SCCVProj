import os
from PIL import Image

def resize_and_convert_images(folder_path):
    """
    Resizes all images in the specified folder to 256x256 pixels.
    Converts images with less than 3 layers (not in RGB mode) to RGB.

    :param folder_path: Path to the folder containing the images.
    """
    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter out non-image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]

    # Iterate over each image
    for image_name in images:
        # Full path of the image
        image_path = os.path.join(folder_path, image_name)

        # Open and process the image
        with Image.open(image_path) as img:
            # Check if the image does not have 3 layers (not RGB)
            if img.mode != 'RGB':
                print(f"{image_name} does not have 3 layers. Converting to RGB.")
                img = img.convert('RGB')

            # Resize the image
            img = img.resize((256, 256))

            # Save the resized and converted image back to the folder
            img.save(image_path)

import os

def convert_jpeg_to_jpg(folder_path):
    """
    Converts all .jpeg files in the specified folder to .jpg

    :param folder_path: Path to the folder containing the images.
    """
    # List all files in the directory
    files = os.listdir(folder_path)

    # Iterate over each file
    for file_name in files:
        # Check if the file is a .jpeg file
        if file_name.lower().endswith('.jpeg'):
            # Define the old and new file paths
            old_file = os.path.join(folder_path, file_name)
            new_file = os.path.join(folder_path, file_name[:-5] + '.jpg')

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Converted '{old_file}' to '{new_file}'")


# Example usage
folder = './locDB'  # Replace with your folder path
#resize_and_convert_images(folder)
convert_jpeg_to_jpg(folder)