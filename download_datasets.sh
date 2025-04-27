#!/bin/bash

# Define the folder to store the downloaded files
DEST_FOLDER="data"

# Create the folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

# List of URLs to download
URLs=("https://dataverse.harvard.edu/api/access/datafile/3478527?gbrecs=true" "https://dataverse.harvard.edu/api/access/datafile/7081897?gbrecs=true" "https://data.mendeley.com/public-files/datasets/9x87km32n6/files/a55dd219-3d6a-46bc-aece-4d961933b351/file_downloaded" "https://dataverse.harvard.edu/api/access/datafile/7568746?format=tab&gbrecs=true" "https://zenodo.org/records/6834906/files/MRI_Biopsy_Data.xls?download=1" "https://figshare.com/ndownloader/files/36346823" "https://data.mendeley.com/public-files/datasets/5sxfpyzmx4/files/adc31fdb-9847-4b83-bb87-bbd9766ea624/file_downloaded" "https://plos.figshare.com/ndownloader/files/17431067" "https://plos.figshare.com/ndownloader/files/2578419" "https://plos.figshare.com/ndownloader/files/10103526" "https://figshare.com/ndownloader/files/36971380" "https://figshare.com/ndownloader/files/30741169")

# Corresponding custom filenames
FILENAMES=("Korea1.pdf" "Korea2.xlsx" "Italy.xlsx" "Spain.tab" "Germany1.xls" "USA1.xlsx" "China1.sav" "Germany2.xlsx" "USA2.xlsx" "Finland.txt" "Turkey.sav" "China2.zip")

# Loop through the URLs and download each file
for i in "${!URLs[@]}"; do
  URL="${URLs[$i]}"
  FILENAME="${FILENAMES[$i]}"

  # Download the file to the destination folder with the custom filename
  echo "Downloading $URL as $FILENAME..."
  wget -O "$DEST_FOLDER/$FILENAME" "$URL"
  
  # Check if download was successful
  if [ $? -eq 0 ]; then
    echo "Downloaded $FILENAME successfully!"
  else
    echo "Failed to download $URL"
  fi
done