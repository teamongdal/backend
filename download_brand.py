import os
import requests
import csv
from urllib.parse import urlparse

# Global set to track downloaded image names (without extension)
downloaded_names = set()

def download_image(url: str, save_dir: str) -> bool:
    try:
        # Skip if URL is "없음"
        if url == "없음":
            print("Skipping image download for URL marked '없음'")
            return False

        # Parse the URL to extract the file name (e.g., "topten.png")
        parsed_url = urlparse(url)
        base_filename = os.path.basename(parsed_url.path)  # e.g., "topten.png"
        # Split the filename to separate name and extension
        name, ext = os.path.splitext(base_filename)

        # Skip if this image name was already downloaded
        if name in downloaded_names:
            print(f"Skipping {name}, already downloaded.")
            return False

        # Define the full path where the image will be saved
        file_path = os.path.join(save_dir, base_filename)

        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Set headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.124 Safari/537.36"
        }

        # Download the image content
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Write the image content to the file
        with open(file_path, 'wb') as file:
            file.write(response.content)

        print(f"Downloaded: {base_filename}")

        # Mark the image as downloaded
        downloaded_names.add(name)
        return True

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_images_from_csv(csv_file: str, save_dir: str):
    # Open and read the CSV file
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Get the image URL from the "brand_image" column
            image_url = row.get("brand_image")
            # Skip if the brand_image is an empty string
            if image_url == "":
                continue
            download_image(image_url, save_dir)

# Usage example:
csv_file = "product_combined.csv"
save_dir = "static/local_data/brand_images/"  # Directory to save the downloaded images
download_images_from_csv(csv_file, save_dir)
