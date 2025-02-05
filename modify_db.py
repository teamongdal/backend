from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import SessionLocal, Video, Favorite, Product, Highlight  # Import the database session and model

# Assuming Video class and SessionLocal are defined elsewhere
def add_video_to_db(vid_name, vid_url, vid_img):
    db = SessionLocal()
    try:
        new_video = Video(
            video_name=vid_name,  # Video name
            video_url=vid_url,  # Video URL (cloud)
            video_image =vid_img # Thumbnail URL (local)
        )
        db.add(new_video)
        db.commit()
        print("Video added successfully!")
    except SQLAlchemyError as e:
        db.rollback()  # Rollback in case of error
        print(f"Error adding video: {str(e)}")
    finally:
        db.close()

# Call the function to add the video
vid_name = "나는 SOLO E151"
vid_url = "https://ai-shop-bucket.s3.ap-southeast-2.amazonaws.com/vod/나는_SOLO_E151_10m.mp4"
vid_img = "C:/backend-skt/local_data/video_thumbnail/2188058 나는_SOLO_E151_Thumbnail.png"
add_video_to_db(vid_name, vid_url, vid_img)