import json
import os

# Function to modify the content of the JSON files
def modify_json_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Modify the content to match the new format
    new_data = {
        "product_name": data["product_name"],
        "brand_name": "MUSINSA",
        "price": data["product_price"],
        "detail_image_url_1": "https://www.musinsa.com/products/3213670",
        "detail_image_url_2": "https://www.musinsa.com/products/3213670",
        "detail_image_url_3": "https://www.musinsa.com/products/3213670"
    }
    
    return new_data

# Specify the paths to the folders on your local machine
product_metadata_folder = r'C:\github\ongdal\backend\local_data\product_metadata'
modified_folder = r'C:\github\ongdal\backend\local_data\modified_product_metadata'

# Create the modified folder if it does not exist
os.makedirs(modified_folder, exist_ok=True)

# List all files in the product_metadata folder
metadata_files = os.listdir(product_metadata_folder)

# Modify all JSON files and save them to the new folder
modified_files = {}
for file_name in metadata_files:
    file_path = os.path.join(product_metadata_folder, file_name)
    modified_content = modify_json_content(file_path)
    
    # Save the modified content
    modified_file_path = os.path.join(modified_folder, file_name)
    with open(modified_file_path, 'w', encoding='utf-8') as modified_file:
        json.dump(modified_content, modified_file, ensure_ascii=False, indent=4)
    
    modified_files[file_name] = modified_file_path

# Output the list of modified files with their paths
modified_files
