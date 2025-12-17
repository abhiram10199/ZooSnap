# extract zip file

import zipfile
import os
def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to the specified directory.

    :param zip_path: Path to the zip file.
    :param extract_to: Directory where the contents will be extracted.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# Example usage
if __name__ == "__main__":
    zip_file_path = 'ZooAnimals-20251207T110314Z-1-002.zip'  # Replace with your zip file path
    output_directory = 'ZooAnimals'  # Replace with your desired output directory
    extract_zip(zip_file_path, output_directory)   