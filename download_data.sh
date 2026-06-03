echo Saving to $(pwd)/data
gdown -O $(pwd)/data/  https://drive.google.com/uc?id=17pd_nyQ69U8ymIdMuGhOlzXYZDB5iAyo # Save expanded_frame.zip
echo Unzipping files...
find $(pwd)/data -type f -name "*.zip" -exec unzip -q -d $(pwd)/data {} \;
echo Done.
echo Folders saved:
ls -d $(pwd)/data/*