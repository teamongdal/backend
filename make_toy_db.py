import os
import json
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import SessionLocal, Video, UserFavorite, Product, Highlight, UserVideo

def add_video_to_db():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_dir = os.path.join(project_dir, "local_data", "video_metadata")
    thumbnail_dir = os.path.join(project_dir, "local_data", "video_thumbnail")

    db = SessionLocal()

    try:
        for filename in os.listdir(metadata_dir):
            if filename.endswith(".json"):
                metadata_path = os.path.join(metadata_dir, filename)
                
                # Read metadata
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                # Extract required fields
                vid_name = metadata.get("video_name", "")
                vid_url = metadata.get("video_url", "")

                # Derive video image path by replacing file extension in metadata filename
                base_name = os.path.splitext(filename)[0]  # Remove .json extension
                possible_extensions = [".jpg", ".png", ".jpeg"]  # Handle multiple formats
                
                vid_img = None
                for ext in possible_extensions:
                    img_path = os.path.join(thumbnail_dir, base_name + ext)
                    if os.path.exists(img_path):  # Check if the file exists
                        vid_img = img_path
                        break  # Stop at the first valid image found
                
                if vid_img is None:
                    print(f"Skipping: No matching thumbnail found for {filename}")
                    continue
                
                # Check if video already exists
                existing_video = db.query(Video).filter(Video.video_url == vid_url).first()
                if existing_video:
                    print(f"Video already exists: {vid_url}")
                    continue
                
                # Create and add new video entry
                new_video = Video(
                    video_name=vid_name,
                    video_url=vid_url,
                    video_image=vid_img  # Use derived thumbnail path
                )
                db.add(new_video)

        db.commit()
        print("Videos added successfully!")

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding videos: {str(e)}")

    finally:
        db.close()

# UPDATED FOR detail_image_url COLUMN
def add_products_to_db():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(project_dir, "local_data", "product_image")
    metadata_dir = os.path.join(project_dir, "local_data", "modified_product_metadata")
    
    db = SessionLocal()
    try:
        for filename in os.listdir(metadata_dir):
            if filename.endswith(".json"):
                metadata_path = os.path.join(metadata_dir, filename)
                image_path = os.path.join(image_dir, filename.replace(".json", ""))  # Get corresponding image path
                
                # Check if image exists
                image_file = None
                for ext in [".jpg", ".png", ".jpeg"]:
                    if os.path.exists(image_path + ext):
                        image_file = image_path + ext
                        break
                
                if not image_file:
                    print(f"Image not found for {filename}")
                    continue
                
                # Read metadata
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                # Extract required fields
                product_name = metadata.get("product_name", "")
                brand_name = metadata.get("brand_name", "")
                price = metadata.get("price", "")
                detail_image_url_1 = metadata.get("detail_image_url_1", "")
                detail_image_url_2 = metadata.get("detail_image_url_2", "")
                detail_image_url_3 = metadata.get("detail_image_url_3", "")

                # Check if product already exists
                existing_product = db.query(Product).filter_by(product_name=product_name, price=price).first()
                if existing_product:
                    print(f"Product already exists: {product_name}")
                    continue
                
                # Create product entry
                new_product = Product(
                    product_image_url=image_file,
                    product_name=product_name,
                    brand_name=brand_name,
                    price=price,
                    detail_image_url_1=detail_image_url_1,
                    detail_image_url_2=detail_image_url_2,
                    detail_image_url_3=detail_image_url_3
                )
                db.add(new_product)
        
        db.commit()
        print("Products added successfully!")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding products: {str(e)}")
    finally:
        db.close()

# UPDATED FOR 'similar_product_list' COLUMN
def add_highlights_db():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    highlight_image_dir = os.path.join(project_dir, "local_data", "highlight_image")
    highlight_metadata_dir = os.path.join(project_dir, "local_data", "highlight_metadata")

    db = SessionLocal()

    try:
        for filename in os.listdir(highlight_image_dir):
            if filename.startswith("highlight_video_") and filename.endswith((".png", ".jpg")):
                format_filename = filename.replace("highlight_video_", "").rsplit(".", 1)[0]
                parts = format_filename.split("_idx_")

                if len(parts) != 2:
                    print(f"Skipping invalid file name: {filename}")
                    continue

                video_id, highlight_idx = map(int, parts)
                highlight_image_url = os.path.join(highlight_image_dir, filename)

                # Find matching metadata file
                metadata_filename = f"highlight_video_{video_id}_idx_{highlight_idx}.json"
                metadata_path = os.path.join(highlight_metadata_dir, metadata_filename)

                if not os.path.exists(metadata_path):
                    print(f"Skipping: No metadata found for {filename}")
                    continue

                # Read metadata JSON
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                product_id = metadata.get("product_id")
                similar_product_1 = metadata.get("similar_product_1")
                similar_product_2 = metadata.get("similar_product_2")
                similar_product_3 = metadata.get("similar_product_3")
                # similar_product_list = metadata.get("similar_product_list", [])
                ## Convert similar product list to JSON string
                # similar_product_json = json.dumps(similar_product_list)

                # Check if product exists in database
                product = db.query(Product).filter(Product.product_id == product_id).first()
                if not product:
                    print(f"Skipping: No matching product_id {product_id} found in database.")
                    continue

                # Check if highlight entry already exists
                existing_highlight = db.query(Highlight).filter(
                    Highlight.video_id == video_id,
                    Highlight.highlight_idx == highlight_idx,
                    Highlight.product_id == product_id
                ).first()

                if existing_highlight:
                    print(f"Skipping: Highlight already exists for video_id={video_id}, highlight_idx={highlight_idx}, product_id={product_id}")
                    continue

                # Create highlight entry
                new_highlight = Highlight(
                    video_id=video_id,
                    highlight_idx=highlight_idx,
                    highlight_image_url=highlight_image_url,
                    product_id=product_id,
                    similar_product_1=similar_product_1,
                    similar_product_2=similar_product_2,
                    similar_product_3=similar_product_3
                    # similar_product_list=similar_product_json  # Store as JSON string
                )

                db.add(new_highlight)
                print(f"Added highlight: video_id={video_id}, highlight_idx={highlight_idx}, product_id={product_id}")

        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error while adding highlights: {str(e)}")
    finally:
        db.close()

def add_uservideo_to_db():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    uservideo_dir = os.path.join(project_dir, "local_data", "user_video")

    db = SessionLocal()
    try:
        for filename in os.listdir(uservideo_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(uservideo_dir, filename)

                # Read user video data
                with open(file_path, "r", encoding="utf-8") as f:
                    user_video_data = json.load(f)

                user_id = user_video_data.get("user_id")
                video_ids = user_video_data.get("video_id", [])

                for video_id in video_ids:
                    # Check if the entry already exists
                    existing_entry = db.query(UserVideo).filter(
                        UserVideo.user_id == user_id,
                        UserVideo.video_id == video_id
                    ).first()

                    if existing_entry:
                        print(f"UserVideo already exists: user_id={user_id}, video_id={video_id}")
                        continue

                    # Create and add new entry
                    new_entry = UserVideo(user_id=user_id, video_id=video_id)
                    db.add(new_entry)

        db.commit()
        print("User videos added successfully!")

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding user videos: {str(e)}")
    finally:
        db.close()

def add_userfavorite_to_db():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    userfavorite_dir = os.path.join(project_dir, "local_data", "user_favorite")

    db = SessionLocal()
    try:
        for filename in os.listdir(userfavorite_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(userfavorite_dir, filename)

                # Read user favorite data
                with open(file_path, "r", encoding="utf-8") as f:
                    user_favorite_data = json.load(f)

                user_id = user_favorite_data.get("user_id")
                product_ids = user_favorite_data.get("product_id", [])

                for product_id in product_ids:
                    # Check if the entry already exists
                    existing_entry = db.query(UserFavorite).filter(
                        UserFavorite.user_id == user_id,
                        UserFavorite.product_id == product_id
                    ).first()

                    if existing_entry:
                        print(f"UserFavorite already exists: user_id={user_id}, product_id={product_id}")
                        continue

                    # Create and add new entry
                    new_entry = UserFavorite(user_id=user_id, product_id=product_id)
                    db.add(new_entry)

        db.commit()
        print("User favorites added successfully!")

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding user favorites: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    add_video_to_db() # add videos to db
    add_products_to_db() # add ALL products (provided) to db
    add_highlights_db() # add highlight scenes (provided) to db
    add_uservideo_to_db() # add user's video to db
    add_userfavorite_to_db() # add user's favorite to db