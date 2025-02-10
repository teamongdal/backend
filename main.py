# backend/main.py using FastAPI

############################################################################################################
############################################################################################################
############################################################################################################

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import shutil
import torch
import cv2
import numpy as np
from pathlib import Path
# from config import CONFIG
# from model import build_model
# from utils import non_max_suppression, scale_coords, xyxy2xywh
# from data import letterbox
from typing import List, Optional
import random
from sqlalchemy.orm import Session
from database import SessionLocal, Video, Favorite, Product, Highlight  # Import the database session and model

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

# reset current SimilarProduct DB 
# similarProduct DB: { rank, product_id }

### 1.(GET) VOD 리스트 페이지 - VOD 리스트 조회 ###
@app.get("/api/video_list")
def get_videos(user_id: int, db: Session = Depends(get_db)):
        videos = db.query(Video).filter(Video.user_id == user_id).all()
        # return [{"video_id": v.video_id, "video_name": v.video_name, "video_url": v.video_url, "video_image": v.video_image} for v in videos]
        return [{"video_id": v.video_id, "video_name": v.video_name, "video_image": v.video_image} for v in videos]

### 2. (GET) VOD 재생 페이지 - 해당 VOD 조회 ###
# example: curl -X 'GET' 'http://127.0.0.1:8000/api/video_fetch?video_id=1'
@app.get("/api/video_play")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.video_id == video_id).first()

    if not video:
        return {"error": "Video not found"}
    
    return {"video_id": video.video_id, "video_name": video.video_name, "video_url": video.video_url}

# TODO: Embedding AI Model for capturing frame and detecting products
# TODO: HOW TO CAPTURE ONE FRAME FROM VIDEO?
### 3.1. (GET) VOD 재생 페이지 - 유저 음성 발화 상품 검색 ###
@app.get("/api/search_product")
def search_product(req_stt: str, img: ?, time: ?, db: Session = Depends(get_db)):
    product_id = search_ai()
    return None # should return product_id from Product DB + call next AI Model

# TODO: AI Model [FAISS Model] for finding similar products based on the detected product (top 5?)
### 3.2. (GET) 
@app.get("/api/recommend_product")
def recommend_product():



### 4.(GET) 상품 검색 결과 페이지 - 상품 리스트 조회 ###
@app.get("/api/product_list")
def get_similar_products(user_id: int, product_id: int, db: Session = Depends(get_db)):
    
    product = db.query(Product).filter(Product.product_id == product_id).first()
    products = db.query(Product).all() # (유사한 옷을 찾는 query)

    if product:
        return [
            {
                "product_id": p.product_id,
                "product_pic_url": p.product_pic_url,
                "brand_name": p.brand_name,
                "product_name": p.product_name,
                "price": p.price,
                "detail": p.detail,
            }
            for p in products
        ]




### (POST) 찜하기 ###
# curl -X 'POST' 'http://127.0.0.1:8000/api/product_like?user_id=1&product_id=123'
@app.post("/api/product_like")
def like_product(user_id: int, product_id: int, db: Session = Depends(get_db)):
    # 찜 할때 이미 찜 했는지 확인하고 추가
    existing_like = db.query(Favorite).filter(
        Favorite.user_id == user_id,
        Favorite.product_id == product_id
    ).first()

    if existing_like:
        return {"message": "Product already liked by the user."}
    
    new_favorite = Favorite(
        user_id=user_id,
        product_id=product_id
    )
    
    db.add(new_favorite)
    db.commit()
    db.refresh(new_favorite)
    
    return {"message": "Product liked successfully!", "favorite_id": new_favorite.id}


### (POST) 찜 삭제 ###
@app.post("/api/product_unlike")
def unlike_product(user_id: int, product_id: int, db: Session = Depends(get_db)):
    # 찜 상품 찾기
    favorite_entry = db.query(Favorite).filter(
        Favorite.user_id == user_id,
        Favorite.product_id == product_id
    ).first()

    # 못찾으면 아무것도 안하기
    if not favorite_entry:
        return {"message": "Product not found in favorites."}
    
    # 찾으면 찜 목록에서 삭제
    db.delete(favorite_entry)
    db.commit()
    
    return {"message": "Product unliked successfully!"}

### (GET) 찜 상품 목록 조회 ###
@app.get("/api/product_like_list")
def get_liked_products(user_id: int, db: Session = Depends(get_db)):
    liked_products = db.query(Favorite.product_id).filter(Favorite.user_id == user_id).all()
    
    product_ids = [product_id[0] for product_id in liked_products]

    # {user_id: 1, liked_products: [1, 2, 3, 4, 5]}
    return {"user_id": user_id, "liked_products": product_ids}


 
# ### (GET) 유사 상품 리스트에서 특정 상품 조회 ### -- 삭제 예정
# @app.get("/api/similar_product_list/{product_id}")
# def dummy():
#     return 0

### (GET) 하이라이트 상품 리스트 조회 ###
@app.get("/api/highlight_product_list")
def get_highlight_products(video_id: int, db: Session = Depends(get_db)):
    highlights = db.query(Highlight).filter(Highlight.video_id == video_id).all()

    if not highlights:
        return {"error": "No highlights found for this video_id"}
    
    return [
        {
            "highlight_idx": h.highlight_idx,
            "highlight_pic_url": h.highlight_pic_url,
            "product_id": h.product_id,  
            "product_pic_url": h.product_pic_url,
            "brand_name": h.brand_name,
            "product_name": h.product_name,
            "price": h.price,
        }
        for h in highlights
    ]


##################### AI MODEL #####################

### (GET) AI 모델 (1) - 현재 프레임에서 옷 검출 ###
@app.get("/api/detect/")
def dummy():
    return 0
# TODO

### (GET) AI 모델 (2) - 옷 검출 결과 사용해서 DB에서 유사 옷 검색 + 추천 리스트 반환 ###
@app.get("/api/recommend/")
def dummy():
    return 0
# TODO

### (POST) Model Sample (Upload Image -> Run AI Detection Model -> Return prediction + result_image) ###
# TODO - REPLACE MODEL
# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     try:
#         # 파일 확장자 검사
#         allowed_extensions = {'.jpg', '.jpeg', '.png'}
#         file_ext = os.path.splitext(file.filename)[1].lower()
        
#         # 디버깅을 위한 로그 추가
#         print(f"Received file: {file.filename}, Extension: {file_ext}")
        
#         if not file_ext:
#             file_ext = '.jpg'  # 확장자가 없는 경우 기본값으로 .jpg 설정
            
#         if file_ext not in allowed_extensions:
#             raise HTTPException(status_code=400, detail="Only JPG, JPEG and PNG files are allowed")

#         # 업로드 디렉토리가 없으면 생성
#         os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
#         # 파일 저장
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # 이미지 처리 및 결과 반환
#         result_img, detections = detector.detect(file_path)
        
#         # 결과 이미지 저장
#         result_filename = f"result_{file.filename}"
#         result_filepath = os.path.join(UPLOAD_FOLDER, result_filename)
#         cv2.imwrite(result_filepath, result_img)
        
#         # 결과 데이터 준비
#         result_data = {
#             'original_image': f"/uploads/{file.filename}",
#             'result_image': f"/uploads/{result_filename}",
#             'detections': detections
#         }
        
#         return JSONResponse(content=result_data, status_code=200)
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


##################### initialize backend server #####################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)