import numpy as np
import os

# Assuming image_arr has shape (10000, 144, 256)
# image_arr = np.load('your_image_array.npy')  # Load your array

image_arr=np.load("image_stacks/imagestack_nov232025.npy")

output_dir = 'image_stacks/dec3_splitup'
os.makedirs(output_dir, exist_ok=True)

batch_size = 100
n_images = image_arr.shape[0]

# Split into batches and save
for batch_idx in range(0, n_images, batch_size):
    end_idx = min(batch_idx + batch_size, n_images)
    
    # Create batch folder
    batch_folder = f'{output_dir}/batch_{batch_idx:05d}_to_{end_idx:05d}'
    os.makedirs(batch_folder, exist_ok=True)
    
    # Slice and save
    batch = image_arr[batch_idx:end_idx]
    np.save(f'{batch_folder}/images_{batch_idx:05d}_to_{end_idx:05d}.npy', batch)
    
    print(f"Saved batch {batch_idx}-{end_idx}: shape {batch.shape}")

print(f"\nAll batches saved to {output_dir}/")
print(f"Total batches: {(n_images + batch_size - 1) // batch_size}")