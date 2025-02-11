# backend/main.py using FastAPI

############################################################################################################
############################################################################################################
############################################################################################################

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
#import os
#import shutil
#import torch
#import cv2
#import numpy as np
from pathlib import Path
# from config import CONFIG
# from model import build_model
# from utils import non_max_suppression, scale_coords, xyxy2xywh
# from data import letterbox
from typing import List, Optional
import random
from sqlalchemy.orm import Session
from database import SessionLocal, Video, Product, Highlight, UserVideo, UserFavorite
import json

# intiaite FastAPI app
app = FastAPI()

# to resolve CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### SQLAlchemy DB ###
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# TODO (think about this)
## reset current SimilarProduct DB 
## similarProduct DB: { rank, product_id }

### 1. (GET) VOD 리스트 페이지 - VOD 리스트 조회 ###
@app.get("/api/video_list")
def get_videos(user_id: int, db: Session = Depends(get_db)):
    # Join UserVideo with Video table to find videos associated with the user
    videos = (db.query(Video).join(UserVideo, Video.video_id == UserVideo.video_id).filter(UserVideo.user_id == user_id).all())

    return [    
        {
            "video_id": v.video_id, 
            "video_name": v.video_name, 
            "video_image": v.video_image
        } 
        for v in videos
    ]

### 2. (GET) VOD 재생 페이지 - 해당 VOD 조회 ###
@app.get("/api/video_play")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.video_id == video_id).first()

    if not video:
        return {"message": "Video not found"}
    
    # return {"video_id": video.video_id, "video_name": video.video_name, "video_url": video.video_url}
    return {"video_url": video.video_url}

# TODO: Embedding AI Model for capturing frame and detecting products
# TODO: Receives audio capture + frame capture from frontend (using stt.py) -> Detection AI Model
# TODO: Check and raise status code for (repeating STT prompt), (no product detected), (no product found in DB) etc.
### 3. (GET) VOD 재생 페이지 - 유저 음성 발화 상품 검색 + 상품 리스트 조회 ###
@app.get("/api/search_product")
def search_product(user_id: int, req_stt: str, img: None, time: None, db: Session = Depends(get_db)):
    return None
#     # TODO (CONFIRM AI MODEL PROCESS)
#     # AI Model 1: LLM 


#     # AI Model 2: Detect product from the frame
#     product_id = search_ai()

#     # AI Model 3 [FAISS]: Recommend similar products using product_id
#     
#     find_product_id = []
#     return get_similar_products(user_id = user_id, find_product_id = find_product_id)
#     !!! the 'return' value will have the format shown below !!!
    # return [
    #     {
    #         "product_id": p.product_id,
    #         "product_image_url": p.product_image_url,
    #         "brand_name": p.brand_name,
    #         "product_name": p.product_name,
    #         "price": p.price,
    #         "detail_image": p.detail_image,
    #         "is_like": p.product_id in favorite_product_ids  # Check if product is liked by the user
    #     }
    #     for p in products
    # ]


### 4. (GET) 상품 검색 결과 페이지 - 상품 리스트 조회 (연동화를 위해 AI 모델 대체하는 함수) ###
## AI 모델에서 get_similar_products 함수 호출 / highlight에서는 "/api/product_list"로 호출
@app.get("/api/product_list")
def get_similar_products(user_id: int, find_product_id: Optional[List[int]] = Query(None), db: Session = Depends(get_db)):
    # Fetch products matching find_product_id
    find_product_id = [1, 3, 15] # sample find_product_id[]
    products = db.query(Product).filter(Product.product_id.in_(find_product_id)).all()
    
    # Fetch favorite products for the given user
    like_product_ids = {fav.product_id for fav in db.query(UserFavorite).filter(UserFavorite.user_id == user_id).all()}
    return {
        "product_list": [
            {
                "product_id": p.product_id,
                "product_image_url": p.product_image_url,
                "brand_name": p.brand_name,
                "product_name": p.product_name,
                "price": p.price,
                "detail_image_url_1": p.detail_image_url_1,  # Returns None if NULL in DB
                "detail_image_url_2": p.detail_image_url_2,
                "detail_image_url_3": p.detail_image_url_3,
                "is_like": p.product_id in like_product_ids  # Check if product is liked by the user
            }
            for p in products
        ]
    }




### 5. (GET) 상품 상세 페이지 - 상품 상세 조회 ### 
# 이건 기존 ("api/product_list")에서 'detail' array에서 이미지 url을 받아와서 보여주기 (frontend 처리)


### 6. (POST) 상품 찜하기 ###
@app.post("/api/product_like")
def product_like(user_id: int, product_id: int, db: Session = Depends(get_db)):
    # 이미 찜되었는지 확인하고 추가
    existing_like = db.query(UserFavorite).filter(
        UserFavorite.user_id == user_id,
        UserFavorite.product_id == product_id
    ).first()

    if existing_like:
        # return {"message": "이미 찜되어있는 상품입니다."}
        return {"success": False}
    
    new_favorite = UserFavorite(
        user_id=user_id,
        product_id=product_id
    )
    
    db.add(new_favorite)
    db.commit()
    db.refresh(new_favorite)
    
    # return {"message": "Product liked successfully!", "favorite_id": new_favorite.id}
    return {"success": True}


### 7. (POST) 상품 찜 삭제 ###
@app.post("/api/product_unlike")
def product_unlike(user_id: int, product_id: int, db: Session = Depends(get_db)):
    # 찜 상품 찾기
    favorite_entry = db.query(UserFavorite).filter(
        UserFavorite.user_id == user_id,
        UserFavorite.product_id == product_id
    ).first()

    # 못찾으면 아무것도 안하기
    if not favorite_entry:
        # return {"message": "Product not found in favorites."}
        return {"sucess": False}
    
    
    # 찾으면 찜 목록에서 삭제
    db.delete(favorite_entry)
    db.commit()
    
    # return {"message": "Product unliked successfully!"}
    return {"success": True}

### 8. (GET) 찜 상품 목록 조회 ###
@app.get("/api/product_like_list")
def product_like_list(user_id: int, db: Session = Depends(get_db)):
    # Join UserFavorite and Product tables to fetch favorite product details
    products = (
        db.query(Product)
        .join(UserFavorite, Product.product_id == UserFavorite.product_id)
        .filter(UserFavorite.user_id == user_id)
        .all()
    )

    return {
        "product_like_list": [
            {
                "product_id": p.product_id,
                "product_image_url": p.product_image_url,
                "brand_name": p.brand_name,
                "product_name": p.product_name,
                "price": p.price,
                "detail_image_url_1": p.detail_image_url_1,  # Returns None if NULL in DB
                "detail_image_url_2": p.detail_image_url_2,
                "detail_image_url_3": p.detail_image_url_3,
            }
            for p in products
        ]
    }




### 9. (GET) 전체 상품 리스트 페이지 - 전체 상품 조회 ###
@app.get("/api/all_product_list")
def get_highlight_products(video_id: int, db: Session = Depends(get_db)):
    highlights = db.query(Highlight).filter(Highlight.video_id == video_id).all()
    
    if not highlights:
        return {"message": "No highlights found for this video_id"}
    
    product_ids = [h.product_id for h in highlights]
    products = {p.product_id: p for p in db.query(Product).filter(Product.product_id.in_(product_ids)).all()}
    
    return {
        "all_product_list": [
            {
                "highlight_idx": h.highlight_idx,
                "highlight_image_url": h.highlight_image_url,
                "product_id": h.product_id,
                "product_image_url": products[h.product_id].product_image_url,
                "brand_name": products[h.product_id].brand_name,
                "product_name": products[h.product_id].product_name,
                "price": products[h.product_id].price,
                "similar_product_1": h.similar_product_1,
                "similar_product_2": h.similar_product_2,
                "similar_product_3": h.similar_product_3
                # "similar_product_list": json.loads(h.similar_product_list) # [product_id of itself + similar products] -> call product_list using this
            }
            for h in highlights
        ]
    }



##################### AI MODEL #####################

### TODO AI 모델 (1) - LLM (Captured Audio + Video Frame 받아서 LLM 처리) / DB에 있는 제일 유사한 옷의 product_id 반환 ###

### TODO AI 모델 (2) - 제일 유사한 옷의 product_id를 받아 / DB에 있는 유사한 옷 5개의 product_id 반환 (아니면 유사 threshold 넘는 상품들) ###

### TODO AI 모델 (3) - ? ###

##################### initialize backend server #####################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)