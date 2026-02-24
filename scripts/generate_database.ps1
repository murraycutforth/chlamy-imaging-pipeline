# Stop script on error
$ErrorActionPreference = "Stop"

conda activate chlamy-impi

# Download latest data
Set-Location ../data
./download_data.ps1

# Generate segmentations
Set-Location ..
$env:PYTHONPATH = (Get-Location).Path
Set-Location chlamy_impi/well_segmentation_preprocessing
python main.py

# Create database
Set-Location ../database_creation
python main.py

# Rename database using today's date
Set-Location ../../output/database_creation
$today = Get-Date -Format "yyyy-MM-dd"
Rename-Item -Path "database.parquet" -NewName "database_$today.parquet"
Rename-Item -Path "database.csv" -NewName "database_$today.csv"
Rename-Item -Path "failed_files.csv" -NewName "failed_files_$today.csv"

Write-Output "Database creation pipeline completed successfully! Yeehah."