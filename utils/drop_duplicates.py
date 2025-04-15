import os
import cv2
import re


def process_data(input_images_folder, output_folder):
    """
    Process images, removing duplicate images (based on hash or name) and saving the results.

    Args:
        input_images_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed data will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Sort files based on their base names (excluding extensions)
    image_files = sorted(os.listdir(input_images_folder), key=lambda f: os.path.splitext(f)[0])
    seen_hashes = set()

    for image_file in image_files:
        image_path = os.path.join(input_images_folder, image_file)

        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping {image_file} due to read error.")
            continue

        # Compute the hash of the image
        image_hash = hash(image.tobytes())

        if image_hash in seen_hashes:
            print(f"Duplicate image found by hash: {image_file}. Skipping.")
            continue

        seen_hashes.add(image_hash)

        # Copy the image to the output folder
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, image)

        print(f"Processed and saved: {image_file}.")


def construct_pattern():
    """
    Construct a pattern to detect duplicates based on examples.

    Returns:
        str: Regular expression pattern for normalization.
    """
    # Updated pattern to match duplicate-like suffixes such as "_1", "(2)", "(3)", etc.,
    # while avoiding valid names like "DSC_0530" or "DSC_0660".
    return r"(_\d+$|\s\(\d+\)$)"


def normalize_name(file_name, pattern):
    """
    Normalize file names to detect duplicates based on patterns.

    Args:
        file_name (str): The file name to normalize.
        pattern (str): Regular expression pattern for normalization.

    Returns:
        str: Normalized file name.
    """
    base_name, _ = os.path.splitext(file_name)
    normalized = re.sub(pattern, "", base_name)
    return normalized


if __name__ == "__main__":
    # Example usage
    input_images_folder = os.path.expanduser("data/staging/distinct_images")
    output_folder = os.path.expanduser("data/staging/distinct_images_v2")
    drop_by = "name"  # Change to "name" to drop duplicates by name

    process_data(input_images_folder, output_folder, drop_by=drop_by)