import os
import shutil

def move_files_by_prefix(source_dir, test_file_path, destination_dir):
    """
    Move files from source_dir to destination_dir based on prefixes listed in test_file_path.

    Args:
        source_dir (str): Path to the source directory to search for files.
        test_file_path (str): Path to the test.txt file containing prefixes.
        destination_dir (str): Path to the destination directory to move files.
    """
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Read prefixes from test.txt
    with open(test_file_path, "r") as file:
        prefixes = [line.strip() for line in file if line.strip()]

    # Recursively find and move matching files
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            for prefix in prefixes:
                if file_name.startswith(prefix):
                    source_path = os.path.join(root, file_name)
                    destination_path = os.path.join(destination_dir, file_name)
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {source_path} -> {destination_path}")
                    break  # Avoid matching the same file with multiple prefixes

if __name__ == "__main__":
    source_dir = "data/processed/v2/images"
    test_file_path = "data/processed/v2/test.txt"
    destination_dir = "data/processed/v2/images/test"

    move_files_by_prefix(source_dir, test_file_path, destination_dir)
