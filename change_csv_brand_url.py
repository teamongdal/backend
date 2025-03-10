import os
import csv
from urllib.parse import urlparse

def update_brand_image_in_csv(csv_file_path: str, images_dir="static/local_data/brand_images/"):
    updated_rows = []
    
    # Ensure images_dir ends with a slash
    if not images_dir.endswith(os.sep):
        images_dir += os.sep
    
    # Open and read the CSV file
    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as infile:
        csv_reader = csv.DictReader(infile)
        fieldnames = csv_reader.fieldnames
        for row in csv_reader:
            image_url = row.get("brand_image", "")
            # Skip if the field is empty or equals "없음"
            if image_url == "" or image_url == "없음":
                updated_rows.append(row)
            else:
                # Parse the URL to extract the file name (e.g., "topten.png")
                parsed_url = urlparse(image_url)
                base_filename = os.path.basename(parsed_url.path)  # e.g., "topten.png"
                # Extract the brand name by removing the extension
                brand_name, _ = os.path.splitext(base_filename)
                
                # Build the new relative path
                new_relative_path = f"static/local_data/brand_images/{brand_name}.png"
                # Convert the relative path to an absolute path to check file existence
                new_abs_path = os.path.join(os.getcwd(), new_relative_path)
                
                # Check if the file exists in the directory
                if os.path.exists(new_abs_path):
                    row["brand_image"] = new_relative_path
                else:
                    print(f"Image not found for brand '{brand_name}', leaving entry unchanged.")
                
                updated_rows.append(row)
    
    # Write the updated rows back to the CSV file
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(updated_rows)
    
    print(f"Updated CSV file: {csv_file_path}")

# Usage example:
csv_file = "product_combined.csv"
update_brand_image_in_csv(csv_file)
