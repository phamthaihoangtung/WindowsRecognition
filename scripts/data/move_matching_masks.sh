images_dir="data\processed\v3\images\val"
masks_dir="data\processed\v3\annotations\train"
dest_masks_dir="data\processed\v3\annotations\val"

python utils/move_matching_masks.py "$images_dir" "$masks_dir" "$dest_masks_dir"