import os
import json
import csv
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import SessionLocal, Video, UserFavorite, Product, Highlight, UserVideo

def transform_image_path(image_path):
    # Check if image_path is "없음"
    if image_path == "없음":
        return image_path
    if image_path:
        # Remove the "C:/ongdal/" prefix from the path
        relative_path = image_path.replace("C:/ongdal/", "")
        # Prepend the IP, port, and static prefix
        return f"static/{relative_path}"
    
def add_video_to_db():
    """
    Process video metadata files named like video_XXXX.json and use the corresponding
    thumbnail from the video_thumbnail folder.
    Assumes metadata has keys: video_id, video_name, video_url, video_image.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_dir = os.path.join(project_dir, "static", "local_data", "video_metadata")

    db = SessionLocal()
    try:
        for filename in os.listdir(metadata_dir):
            # Process only files following the video_XXXX.json pattern
            if not filename.startswith("video_") or not filename.endswith(".json"):
                continue

            metadata_path = os.path.join(metadata_dir, filename)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Extract required fields (all keys match the Video table)
            video_id = metadata.get("video_id")
            if not video_id:
                print(f"Skipping {filename}: 'video_id' not found")
                continue

            # Check if video already exists by video_id
            existing_video = db.query(Video).filter(Video.video_id == video_id).first()
            if existing_video:
                print(f"Video already exists: {video_id}")
                continue

            video_name = metadata.get("video_name", "")
            video_url = metadata.get("video_url", "")
            video_image = metadata.get("video_image", "")

            new_video = Video(
                video_id=video_id,
                video_name=video_name,
                video_url=video_url,
                video_image=video_image
            )
            db.add(new_video)

        db.commit()
        print("Videos added successfully!")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding videos: {str(e)}")
    finally:
        db.close()


    return None

def add_products_to_db():
    """
    Process products from the CSV file (1v_final.csv). Skips lines that begin with '#' (comments).
    Assumes CSV column names match the Product table fields.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(project_dir, "product_combined.csv")
    db = SessionLocal()
    try:
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            # Skip comment lines (lines that start with '#')
            lines = [line for line in csv_file if not line.lstrip().startswith("#")]
            csv_reader = csv.DictReader(lines)
            for row in csv_reader:
                product_code = row.get("product_code")
                if not product_code:
                    print("Skipping row; 'product_code' not found.")
                    continue

                # Check if product already exists
                existing_product = db.query(Product).filter_by(product_code=product_code).first()
                if existing_product:
                    print(f"Product already exists: {product_code}")
                    continue

                new_product = Product(
                    product_code=row.get("product_code"),
                    detail_url=row.get("detail_url"),
                    product_name=row.get("product_name"),
                    product_price=row.get("product_price"),
                    discount_rate=row.get("discount_rate"),
                    final_price=row.get("final_price"),
                    brand_name=row.get("brand_name"),
                    brand_image=row.get("brand_image"),
                    category=row.get("category"),
                    category_sub=row.get("category_sub"),
                    product_images_1=transform_image_path(row.get("product_images_1")),
                    product_images_2=transform_image_path(row.get("product_images_2")),
                    product_images_3=transform_image_path(row.get("product_images_3")),
                    product_images_4=transform_image_path(row.get("product_images_4")),
                    heart_cnt=row.get("heart_cnt"),
                    numof_views=row.get("numof_views"),
                    total_sales=row.get("total_sales"),
                    review_cnt=row.get("review_cnt"),
                    review_rating=row.get("review_rating"),
                    review1=row.get("review1"),
                    review2=row.get("review2"),
                    review3=row.get("review3"),
                    review4=row.get("review4"),
                    review5=row.get("review5"),
                    # gorgeous=row.get("gorgeous"),
                    # TODO CHANGE THIS AFTER RUNNING FAISS
                    # similar_product_1="8seconds_cardigan_0001",
                    # similar_product_2="8seconds_cardigan_0002",
                    # similar_product_3="8seconds_cardigan_0003"
                    # similar_product_1=row.get("similar_product_1"),
                    # similar_product_2=row.get("similar_product_2"),
                    # similar_product_3=row.get("similar_product_3")
                )
                db.add(new_product)

        db.commit()
        print("Products added successfully!")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding products: {str(e)}")
    finally:
        db.close()


def add_highlights_db():
    """
    Process highlight metadata files named like highlight_XXXX_XXXX.json.
    The JSON is assumed to include: video_id, highlight_idx, highlight_image_url, 
    product_code, similar_product_1, similar_product_2, similar_product_3.
    The highlight image filename is joined with the highlight_image folder.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_dir = os.path.join(project_dir, "static", "local_data", "highlight_metadata")
    image_dir = os.path.join(project_dir, "static", "local_data", "highlight_image")
    db = SessionLocal()
    try:
        for filename in os.listdir(metadata_dir):
            if not filename.startswith("highlight_") or not filename.endswith(".json"):
                continue

            metadata_path = os.path.join(metadata_dir, filename)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            video_id = metadata.get("video_id")
            highlight_idx = metadata.get("highlight_idx")
            highlight_product_code= metadata.get("product_code")

            if video_id is None or highlight_idx is None:
                print(f"Skipping {filename}: Missing 'video_id' or 'highlight_idx'")
                continue

            highlight_image_url = metadata.get("highlight_image_url", "")

            new_highlight = Highlight(
                video_id = video_id,
                highlight_idx = highlight_idx,
                highlight_image_url = highlight_image_url,
                product_code = highlight_product_code
                # similar_product_1=metadata.get("similar_product_1", ""),
                # similar_product_2=metadata.get("similar_product_2", ""),
                # similar_product_3=metadata.get("similar_product_3", "")
            )
            db.add(new_highlight)
            # print(f"Added highlight: {filename}")

        db.commit()
        print("Highlights added successfully!")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error while adding highlights: {str(e)}")
    finally:
        db.close()


def add_uservideo_to_db():
    """
    Process user video metadata files named like user_XXXX_video.json.
    Assumes each JSON contains 'user_id' and 'video_id' (which may be a list).
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    uservideo_dir = os.path.join(project_dir, "static", "local_data", "user_video")
    db = SessionLocal()
    try:
        for filename in os.listdir(uservideo_dir):
            if not filename.startswith("user_") or "video" not in filename or not filename.endswith(".json"):
                continue

            file_path = os.path.join(uservideo_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                user_video_data = json.load(f)

            user_id = user_video_data.get("user_id")
            video_ids = user_video_data.get("video_id", [])
            if not isinstance(video_ids, list):
                video_ids = [video_ids]

            for video_id in video_ids:
                existing_entry = db.query(UserVideo).filter(
                    UserVideo.user_id == user_id,
                    UserVideo.video_id == video_id
                ).first()

                if existing_entry:
                    print(f"UserVideo already exists: user_id={user_id}, video_id={video_id}")
                    continue

                new_entry = UserVideo(user_id=user_id, video_id=video_id)
                db.add(new_entry)

        db.commit()
        print("User videos added successfully!")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding user videos: {str(e)}")
    finally:
        db.close()


# def add_userfavorite_to_db():
#     """
#     Process user favorite metadata files named like user_XXXX_favorite.json.
#     Assumes each JSON contains 'user_id' and 'product_code' (which may be a list).
#     """
#     project_dir = os.path.dirname(os.path.abspath(__file__))
#     userfavorite_dir = os.path.join(project_dir, "static", "local_data", "user_favorite")
#     db = SessionLocal()
#     try:
#         for filename in os.listdir(userfavorite_dir):
#             if not filename.startswith("user_") or "favorite" not in filename or not filename.endswith(".json"):
#                 continue

#             file_path = os.path.join(userfavorite_dir, filename)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 user_favorite_data = json.load(f)

#             user_id = user_favorite_data.get("user_id")
#             product_codes = user_favorite_data.get("product_code", [])
#             if not isinstance(product_codes, list):
#                 product_codes = [product_codes]

#             for product_code in product_codes:
#                 existing_entry = db.query(UserFavorite).filter(
#                     UserFavorite.user_id == user_id,
#                     UserFavorite.product_code == product_code
#                 ).first()

#                 if existing_entry:
#                     print(f"UserFavorite already exists: user_id={user_id}, product_code={product_code}")
#                     continue

#                 new_entry = UserFavorite(user_id=user_id, product_code=product_code)
#                 db.add(new_entry)

#         db.commit()
#         print("User favorites added successfully!")
#     except SQLAlchemyError as e:
#         db.rollback()
#         print(f"Error adding user favorites: {str(e)}")
#     finally:
#         db.close()


if __name__ == "__main__":
    add_video_to_db()        # Add videos to db
    add_products_to_db()     # Add products from CSV to db
    add_highlights_db()      # Add highlight scenes to db
    add_uservideo_to_db()    # Add user's videos to db
    # add_userfavorite_to_db() # Add user's favorites to db
