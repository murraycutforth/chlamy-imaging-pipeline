#!/bin/bash
set -e

# Download latest data
cd ../data_13062024
./download_data.sh

# Generate segmentations
cd ..
export PYTHONPATH=$(pwd)
cd chlamy_impi/well_segmentation_preprocessing
python main.py

# Create database
cd ../database_creation
python main.py

# Rename database using today's date
cd ../../output/database_creation
mv database.parquet database_$(date +%Y-%m-%d).parquet
mv database.csv database_$(date +%Y-%m-%d).csv
mv failed_files.csv failed_files_$(date +%Y-%m-%d).csv
