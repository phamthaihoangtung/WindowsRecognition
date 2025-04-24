import os
import shutil

def move_matching_masks(images_dir, masks_dir, dest_masks_dir):
    """
    Find matching masks for images based on base names and move them to a destination folder.

    Args:
        images_dir (str): Directory containing images.
        masks_dir (str): Directory containing masks.
        dest_masks_dir (str): Destination directory for matching masks.
    """
    # Ensure destination directory exists
    os.makedirs(dest_masks_dir, exist_ok=True)

    # Get base names of images
    image_basenames = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))}

    # Iterate through mask files and move matching ones
    for mask_file in os.listdir(masks_dir):
        mask_basename, _ = os.path.splitext(mask_file)
        if mask_basename in image_basenames:
            source_path = os.path.join(masks_dir, mask_file)
            dest_path = os.path.join(dest_masks_dir, mask_file)
            shutil.move(source_path, dest_path)
            print(f"Moved: {source_path} -> {dest_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Move matching masks for images to a destination folder.")
    parser.add_argument("images_dir", help="Directory containing images")
    parser.add_argument("masks_dir", help="Directory containing masks")
    parser.add_argument("dest_masks_dir", help="Destination directory for matching masks")

    args = parser.parse_args()

    move_matching_masks(args.images_dir, args.masks_dir, args.dest_masks_dir)
