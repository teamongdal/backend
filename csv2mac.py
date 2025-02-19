import os
import pandas as pd

def update_product_images(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Define the columns to update
    image_columns = [
        "product_images_1",
        "product_images_2",
        "product_images_3",
        "product_images_4"
    ]
    
    # For each specified column, update the value if it is not "없음"
    for col in image_columns:
        def update_value(x):
            # If the value is "없음", leave it unchanged.
            if x == "없음":
                return x
            # Extract the base file name (without path) and remove its extension.
            base_name = os.path.splitext(os.path.basename(x))[0]
            # Return the new path using the base file name and appending ".jpg"
            return f"/Users/ongdal/local_data/product_combined_images/{base_name}.jpg"
        
        df[col] = df[col].apply(update_value)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)
    print(f"Updated CSV file saved to '{csv_file}'.")

if __name__ == "__main__":
    csv_file = "product_combined.csv"
    update_product_images(csv_file)
