#!/bin/bash

# Specify the download directory
download_dir="data/Munich/2023/raw"

# Create the download directory if it doesn't exist
mkdir -p "$download_dir"

    
links=(
    # Lehel, Munich
    https://download1.bayernwolke.de/a/dop40/data/32692_5335.tif # DOP40
    # https://geodaten.bayern.de/odd_data/laser/692_5335.laz       # LAS
    # https://download1.bayernwolke.de/a/dgm/dgm1/692_5335.tif     # DGM1

    # Old Town, Munich
    https://download1.bayernwolke.de/a/dop40/data/32691_5334.tif # DOP40
    # https://geodaten.bayern.de/odd_data/laser/691_5334.laz       # LAS
    # https://download1.bayernwolke.de/a/dgm/dgm1/691_5334.tif     # DGM1

    # Train Station, Munich
    https://download1.bayernwolke.de/a/dop40/data/32690_5335.tif # DOP40
    # https://geodaten.bayern.de/odd_data/laser/690_5335.laz       # LAS
    # https://download1.bayernwolke.de/a/dgm/dgm1/690_5335.tif     # DGM1
    
    
    https://download1.bayernwolke.de/a/dop40/data/32688_5332.tif
    # forest
    https://download1.bayernwolke.de/a/dop40/data/32692_5347.tif
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
