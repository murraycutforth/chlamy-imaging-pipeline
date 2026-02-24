# Import Rclone - this absolute path is hardcoded to the shared laptop
$rclonePath = 'C:\Users\Burlacot lab\Downloads\rclone-v1.68.1-windows-amd64\rclone-v1.68.1-windows-amd64\rclone.exe'

# Download all files (except xpim) in top-level directory
& $rclonePath "--drive-shared-with-me" "copy" "Google Drive - personal:2023 Screening CliP library/Camera Data (tif, xpim, csv)" "." "--filter" "- /*/**" "--filter" "- /*.xpim" "-v" "--update"
& $rclonePath "--drive-shared-with-me" "copy" "Google Drive - personal:2023 Screening CliP library/Identities of Strains on Plates/Finalized Identities Phase I plates.xlsx" "." "-vv" "--update"

Write-Output "Download complete."

# Write out total number of files
$totalFiles = 0
$extensions = @("*.tif", "*.csv")
foreach ($ext in $extensions) {
    $totalFiles += (Get-ChildItem -Filter $ext).Count
}

$totalTifs = (Get-ChildItem -Filter *.tif).Count

Write-Output "Total number of files: $totalFiles"
Write-Output "Total number of tifs: $totalTifs"

# Delete these random tifs
#Remove-Item -Filter "Copy of*"

# Check that each csv has a corresponding tif with the same filename
foreach ($csv in Get-ChildItem -Filter *.csv) {
    $tif = $csv.Name -replace '\.csv$', '.tif'
    if (-not (Test-Path -Path $tif)) {
        Write-Output "Missing tif file for $($csv.Name), deleting csv file."

        # Remove the csv file
        Remove-Item -Path $csv.FullName
    }
}

# Check that each tif has a corresponding csv with the same filename
foreach ($tif in Get-ChildItem -Filter *.tif) {
    $csv = $tif.Name -replace '\.tif$', '.csv'
    if (-not (Test-Path -Path $csv)) {
        Write-Output "Missing csv file for $($tif.Name), deleting tif file."

        # Remove the tif file
        Remove-Item -Path $tif.FullName
    }
}
