# backend/database.py -> videos.db 생성
# Database using SQLAlchemy -> TODO: MySQL로 변경

############################################################################################################
############################################################################################################
############################################################################################################
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# db name = app_data.db
DATABASE_URL = "sqlite:///./app_data.db"

Base = declarative_base()

# Video Table: video_id / video_name / video_url / video_image
# SAMPLE:
# video_id: 1
# video_name: I_AM_SOLO
# video_url: https://ai-shop-bucket.s3.ap-southeast-2.amazonaws.com/vod/나는_SOLO_E151_10m.mp4
# video_image: local_url 사용
class Video(Base):
    __tablename__ = "videos"
    
    video_id = Column(Integer, primary_key=True, index=True)
    video_name = Column(String, nullable=False)
    video_url = Column(String, nullable=False)
    video_image = Column(String, nullable=False)

# Favorites Table: user_id / product_id
class Favorite(Base):
    __tablename__ = "favorites"
    
    id = Column(Integer, primary_key=True, index=True) # acts as index
    user_id = Column(Integer, index=True)
    product_id = Column(Integer, index=True)

# Product Table: product_id / product_pic_url / brand_name / product_name / price / detail
class Product(Base):
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True, index=True)
    product_pic_url = Column(String, nullable=False)
    brand_name = Column(String, nullable=False)
    product_name = Column(String, nullable=False)
    price = Column(String, nullable=False)
    detail = Column(String, nullable=False)

# Highlight Scene Table: video_id / video_pic_url / product_pic_url / brand_name / product_name / price
class Highlight(Base):
    __tablename__ = "highlights"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, index=True)  # Links to Video Table
    video_pic_url = Column(String, nullable=False)  # Screenshot of the highlight scene
    product_pic_url = Column(String, nullable=False)  # Product image shown in the scene
    brand_name = Column(String, nullable=False)  # Brand of the featured product
    product_name = Column(String, nullable=False)  # Name of the featured product
    price = Column(Integer, nullable=False)  # Price of the featured product

# Database connection
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables (videos and favorites)
Base.metadata.create_all(bind=engine)
