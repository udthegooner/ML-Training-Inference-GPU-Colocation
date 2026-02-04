#!/bin/bash

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="$PROJECT_ROOT/data"

FILE=$1

# Validation for standard CycleGAN datasets
if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" ]]; then
    echo "Available datasets: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

# UPDATED URL: Using the working Berkeley mirror
URL=https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$FILE.zip
ZIP_FILE="$DATA_DIR/$FILE.zip"
TARGET_DIR="$DATA_DIR/$FILE"

mkdir -p "$DATA_DIR"

echo "Downloading $FILE from Berkeley mirror..."
wget -N $URL -O "$ZIP_FILE"

echo "Unpacking dataset..."
unzip -q "$ZIP_FILE" -d "$DATA_DIR"
rm "$ZIP_FILE"

echo "Restructuring to project hierarchy (A/B)..."
mkdir -p "$TARGET_DIR/train/A" "$TARGET_DIR/train/B"
mkdir -p "$TARGET_DIR/test/A" "$TARGET_DIR/test/B"

# Move files from the flat structure to our nested structure
[ -d "$TARGET_DIR/trainA" ] && mv "$TARGET_DIR/trainA"/* "$TARGET_DIR/train/A/" && rm -rf "$TARGET_DIR/trainA"
[ -d "$TARGET_DIR/trainB" ] && mv "$TARGET_DIR/trainB"/* "$TARGET_DIR/train/B/" && rm -rf "$TARGET_DIR/trainB"
[ -d "$TARGET_DIR/testA" ] && mv "$TARGET_DIR/testA"/* "$TARGET_DIR/test/A/" && rm -rf "$TARGET_DIR/testA"
[ -d "$TARGET_DIR/testB" ] && mv "$TARGET_DIR/testB"/* "$TARGET_DIR/test/B/" && rm -rf "$TARGET_DIR/testB"

echo "✅ Success! Dataset is ready at $TARGET_DIR"