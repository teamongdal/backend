import os
import json
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import SessionLocal, Video, Favorite, Product, Highlight

def add_video_to_db(vid_name, vid_url, vid_img):
    db = SessionLocal()
    try:
        # Check if the video already exists based on vid_url
        existing_video = db.query(Video).filter(Video.video_url == vid_url).first()
        if existing_video:
            print(f"Video already exists: {vid_url}")
            return
        
        # Create and add new video entry
        new_video = Video(
            video_name=vid_name,  # Video name
            video_url=vid_url,  # Video URL (cloud)
            video_image=vid_img  # Thumbnail URL (local)
        )
        db.add(new_video)
        db.commit()
        print("Video added successfully!")

    except SQLAlchemyError as e:
        db.rollback()  # Rollback in case of error
        print(f"Error adding video: {str(e)}")
    
    finally:
        db.close()

def add_products_to_db():
    # Directories
    project_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(project_dir, "local_data", "product_images")
    metadata_dir = os.path.join(project_dir, "local_data", "product_metadata")
    
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
                price = metadata.get("product_price", "")
                detail = metadata.get("detail_url", "")
                
                # Check if product already exists
                existing_product = db.query(Product).filter_by(product_name=product_name, price=price, detail=detail).first()
                if existing_product:
                    print(f"Product already exists: {product_name}")
                    continue
                
                # Create product entry
                new_product = Product(
                    product_pic_url=image_file,
                    brand_name="",  # Can be left empty
                    product_name=product_name,
                    price=price,
                    detail=detail
                )
                db.add(new_product)
        
        db.commit()
        print("Products added successfully!")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding products: {str(e)}")
    finally:
        db.close()

def make_highlights_db():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    highlight_dir = os.path.join(project_dir, "local_data", "highlight_video_pic")
    db = SessionLocal()
    
    try:
        for filename in os.listdir(highlight_dir):
            if filename.startswith("highlight_"):
                format_filename = filename.replace("highlight_", "").rsplit(".", 1)[0]
                parts = format_filename.split("_")
                if len(parts) != 3:
                    print(f"Skipping invalid file name: {filename}")
                    continue
                
                video_id, highlight_idx, product_id = map(int, parts)
                highlight_pic_url = os.path.join(highlight_dir, filename)
                
                # Check if product exists
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
                    highlight_pic_url=highlight_pic_url,
                    product_id=product.product_id,
                    product_pic_url=product.product_pic_url,
                    brand_name=product.brand_name,
                    product_name=product.product_name,
                    price=product.price
                )
                
                db.add(new_highlight)
                print(f"Added highlight: video_id={video_id}, highlight_idx={highlight_idx}, product_id={product_id}")
        
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error while adding highlights: {str(e)}")
    finally:
        db.close()


vid_name = "나는 SOLO E151"
vid_url = "https://ai-shop-bucket.s3.ap-southeast-2.amazonaws.com/vod/나는_SOLO_E151_10m.mp4"
vid_img = "C:/github/backend/local_data/video_pic/2188058 나는_SOLO_E151_Thumbnail.png"

# add_video_to_db(vid_name, vid_url, vid_img)
if __name__ == "__main__":
    add_video_to_db(vid_name, vid_url, vid_img)
    add_products_to_db()
    make_highlights_db()