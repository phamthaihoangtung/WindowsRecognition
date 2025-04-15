import os
import shutil
import random

def copy_image_mask_pairs(images_dir, masks_dir, output_images_dir, output_masks_dir, sample_count=None):
    # Normalize paths to use forward slashes
    images_dir = os.path.normpath(images_dir).replace("\\", "/")
    masks_dir = os.path.normpath(masks_dir).replace("\\", "/")
    output_images_dir = os.path.normpath(output_images_dir).replace("\\", "/")
    output_masks_dir = os.path.normpath(output_masks_dir).replace("\\", "/")

    # Check if input directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory does not exist: {masks_dir}")

    # Ensure output directories exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # Get list of image and mask files
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))}
    
    # Find matching pairs
    matching_pairs = [(image_files[base_name], mask_files[base_name]) for base_name in image_files if base_name in mask_files]

    # Sample pairs if sample_count is specified
    if sample_count:
        matching_pairs = random.sample(matching_pairs, min(sample_count, len(matching_pairs)))

    # Copy sampled pairs
    for image_file, mask_file in matching_pairs:
        # Copy image
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_images_dir, image_file))
        # Copy mask
        shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(output_masks_dir, mask_file))
        print(f"Copied: {image_file} and {mask_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copy matching image and mask pairs to separate folders.")
    parser.add_argument("images_dir", help="Directory containing input images")
    parser.add_argument("masks_dir", help="Directory containing input masks")
    parser.add_argument("output_images_dir", help="Directory to store output images")
    parser.add_argument("output_masks_dir", help="Directory to store output masks")
    parser.add_argument("--sample_count", type=int, default=None, help="Number of image-mask pairs to sample (optional)")

    args = parser.parse_args()

    copy_image_mask_pairs(args.images_dir, args.masks_dir, args.output_images_dir, args.output_masks_dir, args.sample_count)
