import os

def rename_images(folder_path, new_name_base, number_format="{:04d}"):
    """
    Renames all images in the specified folder in a sequential order with a specific naming scheme.

    :param folder_path: Path to the folder containing the images.
    :param new_name_base: Base name for the new filenames.
    :param number_format: Format of the numbering part of the new filename.
    """
    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter out non-image files if needed
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # Add or remove extensions as needed
    images = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]

    # Sort the files if needed
    images.sort()  # Sort alphabetically; customize this if you have a different sorting criterion

    # Rename the files
    for idx, image in enumerate(images, start=1):
        # Define the new filename with zero-padded number
        new_name = f"{new_name_base}_{number_format.format(idx)}{os.path.splitext(image)[1]}"

        # Define the full old and new file paths
        old_path = os.path.join(folder_path, image)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed '{old_path}' to '{new_path}'")

# Example usage
folder = './test'  # Replace with your folder path
base_name = 'test'  # Replace with your desired base name
rename_images(folder, base_name)