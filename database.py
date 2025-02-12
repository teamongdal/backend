# backend/database.py -> app_data.db 생성
# database using SQLAlchemy -> TODO: MySQL로 변경(?)

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
    user_id = Column(Integer, index=True)
    video_id = Column(Integer, index=True)

# Video Table: video_id / video_name / video_url / video_image
class Video(Base):
    __tablename__ = "videos"
    
    video_id = Column(Integer, primary_key=True, index=True)
    video_name = Column(String, nullable=False)
    video_url = Column(String, nullable=False)
    video_image = Column(String, nullable=False)

# User's Favorites Table: user_id / product_id
class UserFavorite(Base):
    __tablename__ = "user_favorites"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    product_id = Column(Integer, index=True)

# Product Table: product_id / product_image_url / brand_name / product_name / price / detail_image_url_idx
class Product(Base):
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True, index=True) # 상품 고유번호
    product_image_url = Column(String, nullable=False)         # 상품 이미지
    brand_name = Column(String, nullable=False)                # 브랜드 이름
    product_name = Column(String, nullable=False)              # 상품 이름
    price = Column(String, nullable=False)                     # 상품 가격
    detail_image_url_1 = Column(String, nullable=True)         # 상품 상세 이미지 1
    detail_image_url_2 = Column(String, nullable=True)         # 상품 상세 이미지 2
    detail_image_url_3 = Column(String, nullable=True)         # 상품 상세 이미지 3
    # TODO: 추가 필드 필요할 시 추가
    # 내용링크,상품명,원가격,할인율,할인가격,브랜드,브랜드 이미지,이미지 URL,카테고리 1,카테고리 2,좋아요 수,조회수,누적판매,평점,리뷰수,리뷰1,리뷰2,리뷰3,리뷰4,리뷰5

# Highlight Table: video_id / highlight_idx / highlight_image_url / product_id / similar_product_idx
class Highlight(Base):
    __tablename__ = "highlights"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, index=True)                          # video 필터링
    highlight_idx = Column(Integer, nullable=False)                 # FE에 표시할 순서
    highlight_image_url = Column(String, nullable=False)            # highlight 장면 이미지
    product_id = Column(Integer, index=True)                        # highlight 장면 상품 고유번호
    similar_product_1 = Column(Integer, index=True, nullable=True)  # 유사 상품 고유번호 1
    similar_product_2 = Column(Integer, index=True, nullable=True)  # 유사 상품 고유번호 2
    similar_product_3 = Column(Integer, index=True, nullable=True)  # 유사 상품 고유번호 3
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
