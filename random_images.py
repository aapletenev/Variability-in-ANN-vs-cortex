import os
import numpy as np
from PIL import Image
import random
import pandas as pd
from datetime import datetime

def load_random_images(path_imagenet: str, n: int, train_dir: bool = True, save_images: bool = False, save_path = None):
    """
    Load and process n random images from an ImageNet-like directory structure.
    Returns a NumPy array of images and a DataFrame with image metadata.
    
    Args:
        path_imagenet (str): Path to the ImageNet directory containing subfolders of images.
        n (int): Number of random images to process.
        test_train (bool): Whether or not to use train directory (False would be val), defaults to train.
        save_images (bool): Whether or not to save images to numpy file, defaults to False.
        save_path: Path to save images and metadata to, ensure it is formatted as "folder1/folder2" and NOT "folder1/folder2/" 
    Returns:
        tuple: (np.ndarray, pd.DataFrame)
            - Array of shape (n, 144, 256) containing n grayscale images.
            - DataFrame with columns ['Image Name', 'Folder ID', 'Image ID'] for each image.
    """
    if train_dir: path_imagenet = f'{path_imagenet}//train'
    else: path_imagenet = f'{path_imagenet}//val'

    return_stack = np.zeros((n, 144, 256), dtype=np.uint8)
    
    metadata_df = pd.DataFrame(columns=['Image Name', 'Folder ID', 'Image ID'])
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    all_image_paths = []
    folder_indices = {}
    
    for folder_i, folder in enumerate(os.listdir(path_imagenet)):
        folder_path = os.path.join(path_imagenet, folder)
        if not os.path.isdir(folder_path):
            continue 
        folder_indices[folder] = folder_i
        for image_i, image_name in enumerate(os.listdir(folder_path)):
            if any(image_name.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(folder_path, image_name)
                all_image_paths.append((image_path, folder, image_i))
    
    if len(all_image_paths) < n:
        print(f"Warning: Only {len(all_image_paths)} images available, requested {n}")
        n = len(all_image_paths)
    
    selected_items = random.sample(all_image_paths, n)
    
    metadata_list = []
    for i, (image_path, folder, image_i) in enumerate(selected_items):
        try:
            image = Image.open(image_path)
            image = image.convert('L') 
            if image.size != (256, 144):
                image = image.resize((256, 144))
            
            image_array = np.array(image)
            if image_array.shape != (144, 256):
                print(f"Warning: Image {image_path} has unexpected shape {image_array.shape}")
                continue
            
            return_stack[i] = image_array
            
            metadata_list.append({
                'Image Name': os.path.basename(image_path),
                'Folder ID': folder_indices[folder],
                'Image ID': image_i
            })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{n} images")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    if metadata_list:
        metadata_df = pd.DataFrame(metadata_list)
    
    print(f"We wanted {n} random images and got return stack of shape {return_stack.shape}")

    if save_images:
        if save_path is not None: 
            np.save(f'{save_path}/image_stack({datetime.now().month}-{datetime.now().day}-{datetime.now().year})', return_stack)
            np.save(f'{save_path}/metadata({datetime.now().month}-{datetime.now().day}-{datetime.now().year})', metadata_df)

        else: 
            np.save(f'image_stack({datetime.now().month}-{datetime.now().day}-{datetime.now().year})', return_stack)
            np.save(f'metadata({datetime.now().month}-{datetime.now().day}-{datetime.now().year})', metadata_df)

    return return_stack, metadata_df

# ex
path_imagenet_josh = 'imagenetmini/imagenet-mini'

josh_n = 1000 
images, metadata = load_random_images(path_imagenet = path_imagenet_josh, n = josh_n, 
                                      train_dir = True, save_images = True, save_path="image_stacks/final_image_stack")
print(metadata.head())

