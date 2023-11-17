#!/bin/bash

# Specify the download directory
download_dir="segment-anything/checkpoints"

# Create the download directory if it doesn't exist
mkdir -p "$download_dir"

    
links=(
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth # SAM-ViT-H
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth # SAM-ViT-L
)

# Loop through the array and download each file if it doesn't exist
for link in "${links[@]}"; do
    filename=$(basename "$link")
    filepath="$download_dir/$filename"

    if [ ! -e "$filepath" ]; then
        echo "Downloading $link"
        wget "$link" -P "$download_dir"
    else
        echo "Skipping $link, file already exists"
    fi
done

echo "Downloads completed!"
