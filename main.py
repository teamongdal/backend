# backend/main.py
# using FastAPI

# FRONTEND Design: 
#   Upload Button  - 
#   View Button    - 
#   Restart Button - 

# BACKEND Design:
#   Upload Button -> POST request processed
#       AI Model to take in image and return keywords
#       Database to store uploaded images
#       Perform data-pre-processing
#   View Button -> GET request processed
#   Restart Button -> ?

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
from config import CONFIG
from model import build_model
from utils import non_max_suppression, scale_coords, xyxy2xywh
from data import letterbox
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

# AI Model: Jellyfish Detector example
# class JellyfishDetector:
#     def __init__(self, weights_path, conf_thres=0.5, iou_thres=0.45):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         self.names = [
#             'barrel_jellyfish',
#             'blue_jellyfish',
#             'compass_jellyfish',
#             'lions_mane_jellyfish',
#             'mauve_stinger_jellyfish',
#             'moon_jellyfish'
#         ]
        
#         self.model = build_model()
#         checkpoint = torch.load(weights_path, map_location=self.device)
#         if 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         else:
#             state_dict = checkpoint
#         self.model.load_state_dict(state_dict)
        
#         self.model.to(self.device)
#         self.model.eval()
        
#         self.conf_thres = conf_thres
#         self.iou_thres = iou_thres
#         self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

#     def preprocess_image(self, img_path):
#         img0 = cv2.imread(str(img_path))
#         if img0 is None:
#             raise ValueError(f"Failed to load image: {img_path}")
        
#         img = letterbox(img0, new_shape=640)[0]
#         img = img.transpose(2, 0, 1)
#         img = np.ascontiguousarray(img)
#         img = torch.from_numpy(img).to(self.device)
#         img = img.float()
#         img /= 255.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         return img, img0

#     def detect(self, img_path):
#         img, img0 = self.preprocess_image(img_path)
        
#         with torch.no_grad():
#             pred = self.model(img)
            
#             if isinstance(pred, (tuple, list)):
#                 pred = pred[0]
            
#             pred[..., 4:] = torch.sigmoid(pred[..., 4:])
            
#             pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

#         results = []
#         class_counts = {}
        
#         for i, det in enumerate(pred):
#             if det is not None and len(det):
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
#                 det = det[det[:, 4].argsort(descending=True)]
#                 det = det[:10]
                
#                 for *xyxy, conf, cls in det:
#                     cls_idx = int(cls)
#                     if cls_idx < 0 or cls_idx >= len(self.names):
#                         continue
                    
#                     class_name = self.names[cls_idx]
#                     if class_name not in class_counts:
#                         class_counts[class_name] = 0
#                     if class_counts[class_name] >= 2:
#                         continue
#                     class_counts[class_name] += 1
                    
#                     conf = torch.clamp(conf, 0, 1)
                    
#                     label = f'{class_name} {conf:.2f}'
#                     self.plot_one_box(xyxy, img0, label=label, color=self.colors[cls_idx])
                    
#                     results.append({
#                         'bbox': [coord.item() for coord in xyxy],
#                         'confidence': conf.item(),
#                         'class': class_name
#                     })

#         return img0, results

#     def plot_one_box(self, xyxy, img, color=None, label=None, line_thickness=None):
#         # Plots one bounding box on image img
#         tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
#         color = color or [random.randint(0, 255) for _ in range(3)]
#         c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
#         cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#         if label:
#             tf = max(tl - 1, 1)  # font thickness
#             t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#             c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#             cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#             cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


                        ### REMOVE START ###

# SIMPLE LOCAL DB METHOD -> TODO: upgrade to MySQL

FASHION_DB = "dir_fashion"
os.makedirs(FASHION_DB, exist_ok=True)

VOD_DB = "dir_vod"
os.makedirs(VOD_DB, exist_ok=True)

FAV_DB = "dir_fav"
os.makedirs(FAV_DB, exist_ok=True)

# mount the uploads folder to the app
app.mount("/dir_fashion", StaticFiles(directory="dir_fashion"), name="dir_fashion")
app.mount("/dir_vod", StaticFiles(directory="dir_vod"), name="dir_vod")
app.mount("/dir_fav", StaticFiles(directory="dir_fav"), name="dir_fav")

# initialize AI model and path
base_dir = Path(os.getcwd())
# weights_path = base_dir / 'runs' / 'train' / 'exp' / 'best_model_20250117_221328.pth'
# detector = JellyfishDetector(weights_path)

                        ### REMOVE END ###

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

### (GET) VOD 목록 리스트 ###
@app.get("/api/video_list")
def get_videos(db: Session = Depends(get_db)):
    videos = db.query(Video).all()

    return [{"video_id": v.video_id, "video_name": v.video_name, "video_url": v.video_url, "video_image": v.video_image} for v in videos]

### (GET) VOD 재생 ###
# curl -X 'GET' 'http://127.0.0.1:8000/api/video_fetch?video_id=1'
@app.get("/api/video_fetch")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.video_id == video_id).first()

    if not video:
        return {"error": "Video not found"}
    
    return {"video_name": video.video_name, "video_url": video.video_url}

### (GET) OLD VERSION VOD 재생 ###
# @app.get("/api/video_list/{video_id}")
# def get_video(video_id: int, db: Session = Depends(get_db)):
#     video = db.query(Video).filter(Video.video_id == video_id).first()

#     if not video:
#         return {"error": "Video not found"}
    
#     return {"video_name": video.video_name, "video_url": video.video_url}


### (POST) 찜하기 ###
# curl -X 'POST' 'http://127.0.0.1:8000/api/product_like?user_id=1&product_id=123'
@app.post("/api/product_like")
def like_product(user_id: int, product_id: int, db: Session = Depends(get_db)):
    # Check if the product is already liked by the user
    existing_like = db.query(Favorite).filter(
        Favorite.user_id == user_id,
        Favorite.product_id == product_id
    ).first()

    if existing_like:
        return {"message": "Product already liked by the user."}
    
    # Create a new favorite record
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
    # query all product_id values for the given user_id
    liked_products = db.query(Favorite.product_id).filter(Favorite.user_id == user_id).all()
    
    # convert the list of tuples to a list of product IDs
    product_ids = [product_id[0] for product_id in liked_products]

    return {"user_id": user_id, "liked_products": product_ids}

### (GET) 유사 상품 리스트 조회 ###
@app.get("/api/similar_product_list")
def dummy():
    return 0
# TODO

### (GET) 유사 상품 리스트에서 특정 상품 조회 ###
@app.get("/api/similar_product_list/{product_id}")
def dummy():
    return 0
# TODO 유사 상품에서 nth 상품 불러오기

### (GET) 하이라이트 상품 리스트 조회 ###
@app.get("/api/highlight_product_list")
def dummy():
    return 0


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


# initialize backend server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)