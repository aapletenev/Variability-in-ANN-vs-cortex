import os
import numpy as np
from PIL import Image
import random
import pandas as pd
from datetime import datetime

def load_random_images(path_imagenet: str, n: int, train_dir: bool = True, save_images: bool = False):
    """
    Load and process n random images from an ImageNet-like directory structure.
    Returns a NumPy array of images and a DataFrame with image metadata.
    
    Args:
        path_imagenet (str): Path to the ImageNet directory containing subfolders of images.
        n (int): Number of random images to process.
        test_train (bool): Whether or not to use train directory (False would be val), defaults to train.
        save_images (bool): Whether or not to save images to numpy file, defaults to False.
    Returns:
        tuple: (np.ndarray, pd.DataFrame)
            - Array of shape (n, 144, 256) containing n grayscale images.
            - DataFrame with columns ['Image Name', 'Folder ID', 'Image ID'] for each image.
    """
    if train_dir: path_imagenet = f'{path_imagenet}//train'
    else: path_imagenet = f'{path_imagenet}//val'

    # Initialize return_stack with correct shape for grayscale images
    return_stack = np.zeros((n, 144, 256), dtype=np.uint8)
    
    # Initialize DataFrame for metadata
    metadata_df = pd.DataFrame(columns=['Image Name', 'Folder ID', 'Image ID'])
    
    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Collect all image paths with their folder and image indices
    all_image_paths = []
    folder_indices = {}
    image_indices = {}
    
    for folder_i, folder in enumerate(os.listdir(path_imagenet)):
        folder_path = os.path.join(path_imagenet, folder)
        if not os.path.isdir(folder_path):
            continue  # Skip non-directory files
        folder_indices[folder] = folder_i
        for image_i, image_name in enumerate(os.listdir(folder_path)):
            if any(image_name.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(folder_path, image_name)
                all_image_paths.append((image_path, folder, image_i))
    
    # Check if enough images are available
    if len(all_image_paths) < n:
        print(f"Warning: Only {len(all_image_paths)} images available, requested {n}")
        n = len(all_image_paths)
    
    # Randomly sample n image paths
    selected_items = random.sample(all_image_paths, n)
    
    # Process the selected images and collect metadata
    metadata_list = []
    for i, (image_path, folder, image_i) in enumerate(selected_items):
        try:
            # Open and process the image
            image = Image.open(image_path)
            image = image.convert('L')  # Convert to grayscale
            if image.size != (256, 144):
                image = image.resize((256, 144))
            
            # Convert to NumPy array and verify shape
            image_array = np.array(image)
            if image_array.shape != (144, 256):
                print(f"Warning: Image {image_path} has unexpected shape {image_array.shape}")
                continue
            
            # Store in return_stack
            return_stack[i] = image_array
            
            # Collect metadata
            metadata_list.append({
                'Image Name': os.path.basename(image_path),
                'Folder ID': folder_indices[folder],
                'Image ID': image_i
            })
            
            # Optional progress update
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{n} images")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Create DataFrame from metadata
    if metadata_list:
        metadata_df = pd.DataFrame(metadata_list)
    
    print(f"We wanted {n} random images and got return stack of shape {return_stack.shape}")
    print(f"Metadata DataFrame shape: {metadata_df.shape}")

    if save_images: np.save(f'image_stack({datetime.now().month}-{datetime.now().day}-{datetime.now().year})', return_stack)
    return return_stack, metadata_df

# Example usage
path_imagenet_josh = 'C://Users//joshf//downloads//imagenet-mini'

josh_n = 100  # Number of random images to process
images, metadata = load_random_images(path_imagenet = path_imagenet_josh, n = josh_n, train_dir = True, save_images = True)
print(metadata.head())

