#!/bin/bash

# 1. Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 2. Assume the project root is one level up from the scripts folder
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="$PROJECT_ROOT/data"

FILE=$1

# Validation
if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" ]]; then
    echo "Usage: ./download_cyclegan_data.sh [dataset_name]"
    echo "Available datasets: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE="$DATA_DIR/$FILE.zip"
TARGET_DIR="$DATA_DIR/$FILE"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

echo "Downloading dataset $FILE to $DATA_DIR ..."
wget -N $URL -O "$ZIP_FILE"

echo "Unpacking..."
unzip -q "$ZIP_FILE" -d "$DATA_DIR"
rm "$ZIP_FILE"

echo "Restructuring to project hierarchy..."
# Create the standard A/B structure our datasets.py expects
mkdir -p "$TARGET_DIR/train/A" "$TARGET_DIR/train/B"
mkdir -p "$TARGET_DIR/test/A" "$TARGET_DIR/test/B"

# Move files from the flat Berkeley structure to our nested structure
# Using -n (no-clobber) to prevent errors if directories already exist
[ -d "$TARGET_DIR/trainA" ] && mv "$TARGET_DIR/trainA"/* "$TARGET_DIR/train/A/" && rm -rf "$TARGET_DIR/trainA"
[ -d "$TARGET_DIR/trainB" ] && mv "$TARGET_DIR/trainB"/* "$TARGET_DIR/train/B/" && rm -rf "$TARGET_DIR/trainB"
[ -d "$TARGET_DIR/testA" ] && mv "$TARGET_DIR/testA"/* "$TARGET_DIR/test/A/" && rm -rf "$TARGET_DIR/testA"
[ -d "$TARGET_DIR/testB" ] && mv "$TARGET_DIR/testB"/* "$TARGET_DIR/test/B/" && rm -rf "$TARGET_DIR/testB"

echo "✅ Dataset $FILE is ready at $TARGET_DIR"