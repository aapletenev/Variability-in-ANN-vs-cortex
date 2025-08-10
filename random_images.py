import os
from PIL import Image
import numpy as np

"""
CODE IN DEVELOPMENT
--------------------
"""

def random_images(num, train = True, path = os.path.join("//imagenet-mini")):
    """
    NOTE: directory structure for subfolers as follows
    --imagenet-mini
        -->train
            -->folder1
                -->picture1-1
                -->picture2-1.....
            -->folder2...
            -->folder3...
                -->picture1-3
                -->pictuer2-3
                -->picture....
        -->validation


    Parameters
    ----------
    num: int
        number of images 
    frames: int
        number of frames
    train: bool
        using training or validation set, defaults to train
    path: string
        defaults to cwd for training folder, otherwise path to train/val. folders needed

    Returns
    -------
    folder_id
        folder # used
    image_ids
        image #'s used
    np.stack(final_array)
        stack of num DIFFERENT images in one object, shape (num, 144, 256)
    """

    if train == True: train_val = '//train' 
    else: train_val = '//val'

    folder_path = path + train_val # get path to train/val folders

    visited_folders = [] # keep record of visited folders

    initial_folders = os.listdir(folder_path)

    possible_folders = [i for i in range(np.random.randint(0, len(initial_folders) - 1)) if i not in visited_folders]

    while possible_folders:    
        visited_subfolder

        image_folder_path = folder_path + '//' + initial_folders[folder_id]

        image_folder = os.listdir(image_folder_path) # path to folder in train/val with images

        image_ids = np.random.randint(0,len(image_folder) - 1, size = num)
        final_array = np.empty(shape = (num, 144, 256))

        for i in range(num):
            image_name = '//' + image_folder[i]
            image_path = image_folder_path + image_name
            image = Image.open(image_path)
            image = image.convert('L')
            if image.size != (256, 144): image = image.resize((256, 144))
            image = np.array(image)
            final_array[i] = image

        return visited_folders, image_ids, np.stack(final_array, axis = 0)
