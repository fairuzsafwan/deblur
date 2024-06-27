import bz2
import os

def compress_folder(input_folder, output_archive):
    try:
        # Get list of all files in the input folder and its subdirectories
        files_to_compress = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                files_to_compress.append(os.path.join(root, file))

        # Open the output archive in write-binary mode
        with bz2.open(output_archive, 'wb') as f_out:
            # Compress each file
            for file_to_compress in files_to_compress:
                with open(file_to_compress, 'rb') as f_in:
                    f_out.writelines(f_in)

        print(f"Compression successful. Output archive saved as {output_archive}")
    except Exception as e:
        print(f"Compression failed: {e}")

# Example usage:
input_folder = 'finalFiles'  # Replace with your input folder path
output_archive = 'src.bz2'  # Replace with desired output archive path

compress_folder(input_folder, output_archive)
