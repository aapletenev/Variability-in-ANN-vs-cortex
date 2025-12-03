import numpy as np
import os
import traceback
from prediction import make_predictions

"""
Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
"""
#image_stack = np.load('image_stacks/11-23-2025/imagestack_nov232025.npy')  USE FOR CLUSTERING
image_stack = None
gray_frames = np.full((1, 144, 256), 128)
image_stack = np.concatenate((gray_frames, image_stack), 0)

# Parameters
# BATCHES OCCUR IN MULTIPLES OF 100
start_batch = 0   # Starting batch index (0-based)
end_batch = 50     # Ending batch index (exclusive), None for all remaining batches

image_stacks_dir = 'image_stacks/dec3_splitup'
output_dir = 'predictions/incremental_saves'
#output_dir = 'predictions_folder/december3' USE FOR CLUSTER CODE

os.makedirs(output_dir, exist_ok=True)

# Checkpoint file to track completed batches (resumable on HPC)
checkpoint_path = os.path.join(output_dir, 'completed_batches.txt')
completed = set()
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'r') as f:
        completed = {line.strip() for line in f if line.strip()}

# Gather batch folders
batch_folders = sorted([f for f in os.listdir(image_stacks_dir)
                        if os.path.isdir(os.path.join(image_stacks_dir, f))])

# Select range
selected_batches = batch_folders[start_batch:] if end_batch is None else batch_folders[start_batch:end_batch]
print(f"Processing batches {start_batch} to {end_batch if end_batch else len(batch_folders)}")
print(f"Total: {len(selected_batches)} batches")

def atomic_save_npy(final_path: str, array: np.ndarray):
    """Write array atomically: save to .tmp then replace."""
    tmp_path = final_path + '.tmp'
    np.save(tmp_path, array)
    # os.replace is atomic on both Windows and POSIX
    os.replace(tmp_path, final_path)

def append_checkpoint(batch_id: str):
    """Append a completed batch ID to the checkpoint."""
    with open(checkpoint_path, 'a') as f:
        f.write(batch_id + '\n')

# Process each batch independently and robustly
for batch_idx, batch_folder in enumerate(selected_batches, start=start_batch):
    batch_path = os.path.join(image_stacks_dir, batch_folder)
    output_batch_folder = os.path.join(output_dir, batch_folder)
    os.makedirs(output_batch_folder, exist_ok=True)

    # Skip if already completed (useful for SLURM retries)
    if batch_folder in completed:
        print(f"[Batch {batch_idx}] Skipping already completed: {batch_folder}")
        continue

    try:
        # Find .npy in batch folder
        npy_files = [f for f in os.listdir(batch_path) if f.endswith('.npy')]
        if not npy_files:
            print(f"[Batch {batch_idx}] No .npy file found in {batch_folder}, skipping...")
            append_checkpoint(batch_folder)  # Record as completed to avoid re-processing
            continue

        image_batch_path = os.path.join(batch_path, npy_files[0])
        image_batch = np.load(image_batch_path)

        print(f"[Batch {batch_idx}] Processing {batch_folder}: shape {image_batch.shape}")

        # Predict
        pred_arrs = make_predictions(
            'dynamic', image_batch, 3, [[4, 7]],
            stochastic_bin_param=False,
            noise_seeds=2
        )

        # Save predictions atomically
        final_pred_path = os.path.join(output_batch_folder, 'predictions.npy')
        atomic_save_npy(final_pred_path, pred_arrs[0])
        print(f"[Batch {batch_idx}] Saved predictions to {final_pred_path}")

        # Mark checkpoint
        append_checkpoint(batch_folder)

    except Exception as e:
        # Log error and continue to next batch
        err_log = os.path.join(output_batch_folder, 'error.log')
        with open(err_log, 'w') as f:
            f.write(f"Batch idx: {batch_idx}\nFolder: {batch_folder}\n")
            f.write(f"Error: {repr(e)}\n\n")
            f.write(traceback.format_exc())
        print(f"[Batch {batch_idx}] ERROR. Logged to {err_log}. Continuing...")

print(f"\nAll attempted. Outputs in {output_dir}/")
print(f"Processed batches {start_batch} to {start_batch + len(selected_batches) - 1}")
# ...existing code...