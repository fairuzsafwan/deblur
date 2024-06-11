# Create a clean copy of all images in the directory
mkdir -p cleaned_images
for img in dataset/iss_sharp/*; do
    # Only process actual image files
    if [[ $img == *.jpg || $img == *.jpeg || $img == *.png || $img == *.bmp || $img == *.tiff || $img == *.gif ]]; then
        sips --setProperty format jpeg "$img" --out "cleaned_images/$(basename "$img")"
    fi
done