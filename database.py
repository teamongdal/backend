# backend/database.py -> app_data.db 생성
# database using SQLAlchemy
############################################################################################################
############################################################################################################
############################################################################################################
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json

# db name = app_data.db
DATABASE_URL = "sqlite:///./app_data.db"

Base = declarative_base()

# User's Video Table: user_id / video_id
class UserVideo(Base):
    __tablename__ = "user_videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    video_id = Column(String, nullable=False)

# Video Table: video_id / video_name / video_url / video_image
class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, unique=True, nullable=False)
    video_name = Column(String, nullable=False)
    video_url = Column(String, nullable=False)
    video_image = Column(String, nullable=False)

# User's Favorites Table: user_id / product_code
class UserFavorite(Base):
    __tablename__ = "user_favorites"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    product_code = Column(String, nullable=False)

# Product Table: product_code / product_image_url / brand_name / product_name / price / detail_image_url_idx
class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)          # Unique table index ID
    product_code = Column(String, unique=True, nullable=False)  # Unique product code
    detail_url = Column(String, nullable=False)                 # Detail URL of the product
    product_name = Column(String, nullable=False)               # Product name
    product_price = Column(String, nullable=False)        # Original product price
    discount_rate = Column(String, nullable=True)          # Discount rate (if any)
    final_price = Column(String, nullable=False)          # Final price after discount
    brand_name = Column(String, nullable=False)                 # Brand name
    brand_image = Column(String, nullable=True)                 # URL to the brand image
    category = Column(String, nullable=True)                    # Primary category
    category_sub = Column(String, nullable=True)                # Sub-category
    product_images_1 = Column(String, nullable=True)            # Product image 1 URL
    product_images_2 = Column(String, nullable=True)            # Product image 2 URL
    product_images_3 = Column(String, nullable=True)            # Product image 3 URL
    product_images_4 = Column(String, nullable=True)            # Product image 4 URL
    heart_cnt = Column(String, nullable=True)                 # Count of hearts/likes
    numof_views = Column(String, nullable=True)               # Number of views
    total_sales = Column(String, nullable=True)               # Total sales count
    review_cnt = Column(Text, nullable=True)                # Number of reviews
    review_rating = Column(String, nullable=True)         # Average review rating
    review1 = Column(Text, nullable=True)                       # Review text 1
    review2 = Column(Text, nullable=True)                       # Review text 2
    review3 = Column(Text, nullable=True)                       # Review text 3
    review4 = Column(Text, nullable=True)                       # Review text 4
    review5 = Column(Text, nullable=True)                       # Review text 5
    gorgeous = Column(Integer, nullable=True)                 # Gorgeous rating
    similar_product_1 = Column(String, nullable=True)           # Similar product 1
    similar_product_2 = Column(String, nullable=True)           # Similar product 2
    similar_product_3 = Column(String, nullable=True)           # Similar product 3

# Highlight Table: video_id / highlight_idx / highlight_image_url / product_id / similar_product_idx
class Highlight(Base):
    __tablename__ = "highlights"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, nullable=False)                         # video 필터링
    highlight_idx = Column(Integer, nullable=False)                 # FE에 표시할 순서
    highlight_image_url = Column(String, nullable=False)            # highlight 장면 이미지
    product_code = Column(String, nullable=False)                   # highlight 장면 상품 고유번호
    # similar_product_1 = Column(String, nullable=True)               # 유사 상품 고유번호 1
    # similar_product_2 = Column(String, nullable=True)               # 유사 상품 고유번호 2
    # similar_product_3 = Column(String, nullable=True)               # 유사 상품 고유번호 3
    
    # similar_product_list = Column(Text, nullable=False)           # Store as JSON string / List of product_ids of similar products
    # product_image_url = Column(String, nullable=False)
    # brand_name = Column(String, nullable=False)
    # product_name = Column(String, nullable=False) 
    # price = Column(Integer, nullable=False)
    # detail_image = Column(String, nullable=False)

# Database connection
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables (videos and favorites)
Base.metadata.create_all(bind=engine)
