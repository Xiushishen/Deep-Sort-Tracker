#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

TARGET_DIR="$SCRIPT_DIR/deep_sort/deep/"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Path is not available"
    exit 1
fi

curl -L -o ~/Downloads/archive.zip \
https://www.kaggle.com/api/v1/datasets/download/pengcw1/market-1501

if [ $? -ne 0 ]; then
    echo "Download failed. Please check internet connection and download link"
fi

unzip ~/Downloads/archive.zip -d "$TARGET_DIR"

if [ $? -ne 0 ]; then
    echo "Please check zip file"
    exit 1
fi

mv "$TARGET_DIR/Market-1501-v15.09.15" "$TARGET_DIR/Market1501"

echo "Market-1501 Dataset is downloaded successfully"