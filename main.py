# backend/main.py using FastAPI

############################################################################################################
############################################################################################################
############################################################################################################
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Form, WebSocket, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles
import os
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
# import random
from sqlalchemy.orm import Session
from database import SessionLocal, Video, Product, Highlight, UserVideo, UserFavorite
import json
import io
from stt_backend import process_audio_file
from llm.speech_parser import parse_speech_to_json, classify_speech_request
import subprocess

import asyncio
import websockets
from google.cloud import speech
import socketio

from vod_to_shop import run_vod_to_shop
import ssl

# main.py runs on local machine (localhost:8000)
# TODO host online to allow connection from frontend (Azure: app-container? 느낌)

# intiaite FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# to resolve CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ip = "http://192.168.10.105"
# ip = "http://172.20.213.45"
ip = "http://172.23.150.43"
# ip = "localhost"

def build_image_url(relative_path: str) -> str:
    if relative_path == "없음":
        return relative_path
    base_url = f"{ip}:8000/"
    return f"{base_url}{relative_path}"

def build_image_url_list(relative_paths: list) -> list:
    base_url = f"{ip}:8000/"
    return [
        path if path == "없음" else f"{base_url}{path}"
        for path in relative_paths if path
    ]


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
def get_videos(user_id: str, db: Session = Depends(get_db)):
    # Join UserVideo with Video table to find videos associated with the user
    videos = (db.query(Video).join(UserVideo, Video.video_id == UserVideo.video_id).filter(UserVideo.user_id == user_id).all())

    return [    
        {
            "video_id": v.video_id, 
            "video_name": v.video_name, 
            "video_image": build_image_url(v.video_image)
        } 
        for v in videos
    ]


### 2. (GET) VOD 재생 페이지 - 해당 VOD 조회 ###
@app.get("/api/video_play")
def get_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.video_id == video_id).first()

    if not video:
        return {"message": "Video not found"}
    
    # return {"video_id": video.video_id, "video_name": video.video_name, "video_url": video.video_url}
    return {"video_url": video.video_url, "video_name": video.video_name}


### 3. (POST) VOD 재생 페이지 - 유저 음성 발화 상품 검색 + 상품 리스트 조회 ###
# @app.post("/api/search_product")
# async def search_product(
#     user_id: str,
#     timestamp: str,
#     audio: UploadFile = File(...),  # Audio file
#     image: UploadFile = File(...),  # Image file
#     db: Session = Depends(get_db)  # Database session
# ):
#     try:
#         # Ensure both files were sent
#         if not audio or not image:
#             return {"success": False, "message": "No audio or image file received"}

#         # Validate audio file format: only .wav is accepted.
#         if not audio.filename.lower().endswith(".wav"):
#             return {"success": False, "message": "Invalid audio format. Only .wav files are accepted."}

#         # Validate image file format (jpg, jpeg, png)
#         allowed_image_extensions = (".jpg", ".jpeg", ".png")
#         if not image.filename.lower().endswith(allowed_image_extensions):
#             return {"success": False, "message": "Invalid image format"}

#         # Read audio file into memory and create a file-like object.
#         audio_bytes = await audio.read()
#         audio_buffer = io.BytesIO(audio_bytes)

#         # Read and save the image file to a stable location.
#         image_bytes = await image.read()
#         image_file_path = f"uploaded_{image.filename}"
#         async with aiofiles.open(image_file_path, "wb") as out_file:
#             await out_file.write(image_bytes)
        
#         # If you ever want to support webm conversion, you can convert here.
#         # For now, since we only accept .wav, we use the audio_buffer as-is.
#         wav_buffer = audio_buffer

#         ### STT Begin ###
#         # Process the WAV file for speech-to-text (STT)
#         transcribed_text = process_audio_file(wav_buffer)  # e.g., returns "왼쪽 옷 정보 알려줘"
#         ### STT End ###

#         ### LLM Begin ###
#         # Parse the transcribed text using an LLM
#         llm_keywords = parse_speech_to_json(transcribed_text)
#         llm_keywords_dict = json.loads(llm_keywords)

#         # Check if it's a clothing request; if not, prompt the user to try again.
#         if not llm_keywords_dict.get("is_clothing_request", False):
#             print(llm_keywords_dict.get("speech_text", ""))
#             return {
#                 "success": False,
#                 "message": "다시 말해주세요!",
#                 "user_prompt_original": llm_keywords_dict.get("speech_text", "")
#             }
        
#         print("")
#         print("transcribed_text: ", transcribed_text)
#         print("llm_keywords_dict: ", llm_keywords_dict)
#         print("")
#         print("timestamp: ", timestamp)

#         # Retrieve the location from parsed keywords.
#         location = llm_keywords_dict.get("features", {}).get("location")
#         if location not in ["left", "middle", "right"]:
#             location = "middle"  # Default to "middle" if location is not specified.
#         ### LLM End ###
        
#         ### Detection Model Begin ###
#         if timestamp
#         image_file_path = "static/local_data/highlight_image/highlight_0001_0001.png" # DEFAULT IMAGE <- TODO using timestamp

#         receive_list = run_vod_to_shop(image_file_path, location)

#         ### Detection Model End ###
        
#         print("")
#         print("Myeongpum Code: ", receive_list[0])
#         print("Similar Product List: ", receive_list[1:])
#         print("")

#         # display_list = [myeongpum_code] + similar_product_list

#         products = db.query(Product).filter(Product.product_code.in_(receive_list[1:])).all()
        
#         # Fetch favorite products for the given user
#         like_product_codes = {fav.product_code for fav in db.query(UserFavorite).filter(UserFavorite.user_id == user_id).all()}
        
#         return {
#             "success": True,
#             "message": "Search Product Return Success",
#             "user_id": user_id,
#             "speech_text": transcribed_text,
#             "llm_keywords": llm_keywords_dict,
#             "product_list": [
#                 {
#                     "product_code": p.product_code,
#                     "detail_url": p.detail_url,
#                     "product_name": p.product_name,
#                     "product_price": p.product_price,
#                     "discount_rate": p.discount_rate,
#                     "final_price": p.final_price,
#                     "brand_name": p.brand_name,
#                     "brand_image": build_image_url(p.brand_image),
#                     "category": p.category,
#                     "category_sub": p.category_sub,
#                     "product_images": build_image_url_list([p.product_images_1, p.product_images_2, p.product_images_3, p.product_images_4]),
#                     "heart_cnt": p.heart_cnt,
#                     "numof_views": p.numof_views,
#                     "total_sales": p.total_sales,
#                     "review_cnt": p.review_cnt,
#                     "review_rating": p.review_rating,
#                     "reviews": [p.review1, p.review2, p.review3, p.review4, p.review5],
#                     # "similar_products": [p.similar_product_1, p.similar_product_2, p.similar_product_3],
#                     "is_like": p.product_code in like_product_codes  # Check if product is liked by the user
#                 }
#                 for p in products
#             ]
#         }

#     except Exception as e:
#         return {"success": False, "message": str(e)}

### 3. (POST) VOD 재생 페이지 - 유저 음성 발화 상품 검색 + 상품 리스트 조회 ###
@app.post("/api/search_product")
async def search_product(
    user_id: str,
    time: str,
    video_id: str,
    audio: UploadFile = File(...),  # Audio file
    db: Session = Depends(get_db)  # Database session
):
    try:
        # Ensure both files were sent
        if not audio:
            return {"success": False, "message": "No audio or image file received"}

        # Validate audio file format: only .wav is accepted.
        if not audio.filename.lower().endswith(".wav"):
            return {"success": False, "message": "Invalid audio format. Only .wav files are accepted."}

        # Validate image file format (jpg, jpeg, png)
        # allowed_image_extensions = (".jpg", ".jpeg", ".png")
        # if not image.filename.lower().endswith(allowed_image_extensions):
        #     return {"success": False, "message": "Invalid image format"}

        # Read audio file into memory and create a file-like object.
        audio_bytes = await audio.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Read and save the image file to a stable location.
        # image_bytes = await image.read()
        # image_file_path = f"uploaded_{image.filename}"
        # async with aiofiles.open(image_file_path, "wb") as out_file:
        #     await out_file.write(image_bytes)
        
        # If you ever want to support webm conversion, you can convert here.
        # For now, since we only accept .wav, we use the audio_buffer as-is.
        wav_buffer = audio_buffer

        ### STT Begin ###
        # Process the WAV file for speech-to-text (STT)
        transcribed_text = process_audio_file(wav_buffer)  # e.g., returns "왼쪽 옷 정보 알려줘"
        ### STT End ###

        ### LLM Begin ###
        # Parse the transcribed text using an LLM
        llm_keywords = parse_speech_to_json(transcribed_text)
        llm_keywords_dict = json.loads(llm_keywords)

        # Check if it's a clothing request; if not, prompt the user to try again.
        if not llm_keywords_dict.get("is_clothing_request", False):
            print(llm_keywords_dict.get("speech_text", ""))
            return {
                "success": False,
                "message": "다시 말해주세요!",
                "user_prompt_original": llm_keywords_dict.get("speech_text", "")
            }
        
        print("")
        print("transcribed_text: ", transcribed_text)
        print("llm_keywords_dict: ", llm_keywords_dict)
        print("")
        print("timestamp: ", time)

        # Retrieve the location from parsed keywords.
        location = llm_keywords_dict.get("features", {}).get("location")
        if location not in ["left", "middle", "right"]:
            location = "middle"  # Default to "middle" if location is not specified.
        ### LLM End ###
        
        ### Detection Model Begin ###
        timestamp_float = float(time)

        image_file_path = None

        if video_id == "video_0001":
            # Define the intervals for dami and nj
            dami = [(0.00, 7.07), (8.20, 12.05), (14.15, 17.00), (19.15, 22.05), (25.14, 26.29), (29.21, 31.27), (34.25, 36.15)] # 14.5 ~ 16.19
            nj = [(17.01, 19.14), (22.05, 25.00), (33.11, 34.25), (39.28, 41.17)] # 22.0 ~ 24.5
            
            # Check each interval in dami
            for interval in dami:
                if interval[0] < timestamp_float < interval[1]:
                    image_file_path = "static/local_data/highlight_image/highlight_0001_0001.png"
                    break

            # If no dami interval was matched, check the nj intervals
            if image_file_path is None:
                for interval in nj:
                    if interval[0] < timestamp_float < interval[1]:
                        image_file_path = "static/local_data/highlight_image/highlight_0001_0002.png"
                        break

        elif video_id == "video_0002":
            # Define the intervals for choi and guy
            choi = [(6.17, 12.01), (15.07, 16.13)]
            guy = [(16.15, 25.25), (27.03, 28.01)]
            
            # Check each interval in choi
            for interval in choi:
                if interval[0] < timestamp_float < interval[1]:
                    image_file_path = "static/local_data/highlight_image/highlight_0002_0001.png"
                    break

            # If no choi interval was matched, check the guy intervals
            if image_file_path is None:
                for interval in guy:
                    if interval[0] < timestamp_float < interval[1]:
                        image_file_path = "static/local_data/highlight_image/highlight_0002_0002.png"
                        break

        if image_file_path == None:
            return {
                "success": False,
                "message": "옷 정보 없음!",
                "user_prompt_original": llm_keywords_dict.get("speech_text", "")
            }

        receive_list = run_vod_to_shop(image_file_path, location)

        ### Detection Model End ###
        
        print("")
        print("Myeongpum Code: ", receive_list[0])
        print("Similar Product List: ", receive_list[1:])
        print("")

        # display_list = [myeongpum_code] + similar_product_list

        products = db.query(Product).filter(Product.product_code.in_(receive_list[1:])).all()
        
        # Fetch favorite products for the given user
        like_product_codes = {fav.product_code for fav in db.query(UserFavorite).filter(UserFavorite.user_id == user_id).all()}
        
        return {
            "success": True,
            "message": "Search Product Return Success",
            "user_id": user_id,
            "speech_text": transcribed_text,
            "llm_keywords": llm_keywords_dict,
            "product_list": [
                {
                    "product_code": p.product_code,
                    "detail_url": p.detail_url,
                    "product_name": p.product_name,
                    "product_price": p.product_price,
                    "discount_rate": p.discount_rate,
                    "final_price": p.final_price,
                    "brand_name": p.brand_name,
                    "brand_image": build_image_url(p.brand_image),
                    "category": p.category,
                    "category_sub": p.category_sub,
                    "product_images": build_image_url_list([p.product_images_1, p.product_images_2, p.product_images_3, p.product_images_4]),
                    "heart_cnt": p.heart_cnt,
                    "numof_views": p.numof_views,
                    "total_sales": p.total_sales,
                    "review_cnt": p.review_cnt,
                    "review_rating": p.review_rating,
                    "reviews": [p.review1, p.review2, p.review3, p.review4, p.review5],
                    # "similar_products": [p.similar_product_1, p.similar_product_2, p.similar_product_3],
                    "is_like": p.product_code in like_product_codes  # Check if product is liked by the user
                }
                for p in products
            ]
        }

    except Exception as e:
        return {"success": False, "message": str(e)}

### 3.5 .webm -> .wav ### - REMOVE FOR IOS UPDATE
def convert_webm_to_wav(audio_buffer: io.BytesIO) -> io.BytesIO:
    """
    Converts a WebM audio file (in a BytesIO buffer) to WAV format using FFmpeg.
    Returns a BytesIO buffer containing the WAV data.
    """
    # Ensure the input buffer is at the start
    audio_buffer.seek(0)
    
    # ffmpeg_executable = r"C:\ffmpeg\bin\ffmpeg.exe"

    # FFmpeg command:
    # -i pipe:0  --> read input from stdin
    # -f wav     --> output format: WAV
    # pipe:1     --> write output to stdout
    command = [
        'ffmpeg',
        '-hide_banner',       # Suppress banner output
        '-loglevel', 'error', # Show only errors
        '-i', 'pipe:0',
        '-f', 'wav',
        'pipe:1'
    ]
    
    # Run FFmpeg via subprocess, sending the WebM data via stdin
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Read from the input buffer and get output and errors from FFmpeg
    output, error = process.communicate(audio_buffer.read())
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {error.decode('utf-8')}")
    
    # Create a new BytesIO buffer for the WAV data
    wav_buffer = io.BytesIO(output)
    wav_buffer.seek(0)
    
    return wav_buffer

### 4. (GET) 상품 검색 결과 페이지 - 상품 리스트 조회 (연동화를 위해 AI 모델 대체하는 함수) ###
## (1) AI 모델에서 product_list(user_id, find_product_code) 함수 호출
## (2) highlight에서는 "/api/product_list (user_id, find_product_code)"로 호출
@app.get("/api/product_list")
def product_list(user_id: str, product_code: str, db: Session = Depends(get_db)):
# def product_list(user_id: str, product_code: Optional[str] = Query(None), find_product_code: Optional[List[str]] = Query(None), db: Session = Depends(get_db)):
    # Fetch products matching find_product_code
    #TODO cancel! if (product_id) -> Query on highlights DB for similar_products_id and set this to find_product_id
    #TODO cancel! if (find_product_id) -> do nothing.
    #TODO cancel! optimization: check if product_code in highlights DB before running AI model

    # if product_code:
    #     find_product_code = [product_code] + find_product_code
    # else: # dummy find_product_code
    #     find_product_code = ["blazer_0009", "blazer_0012", "blazer_0015", "blazer_0016"] # [product_code + similar_product_code]
    
    # REMOVE
    if not product_code:
        print("No product code found")
        display_list = ["blazer_0009", "blazer_0012", "blazer_0015", "blazer_0016"] # [product_code + similar_product_code]
    # END REMOVE

    product_entry = db.query(Product).filter(Product.product_code == product_code).first()
    # Initialize the find_product_code list
    display_list = []

    # If a matching entry is found, retrieve similar product codes
    if product_entry:
        display_list = [
            product_code,
            product_entry.similar_product_1,
            product_entry.similar_product_2,
            product_entry.similar_product_3
        ]

    products = db.query(Product).filter(Product.product_code.in_(display_list)).all()
    
    # Fetch favorite products for the given user
    like_product_codes = {fav.product_code for fav in db.query(UserFavorite).filter(UserFavorite.user_id == user_id).all()}
    return {
        "product_list": [
            {
                "product_code": p.product_code,
                "detail_url": p.detail_url,
                "product_name": p.product_name,
                "product_price": p.product_price,
                "discount_rate": p.discount_rate,
                "final_price": p.final_price,
                "brand_name": p.brand_name,
                "brand_image": build_image_url(p.brand_image),
                "category": p.category,
                "category_sub": p.category_sub,
                "product_images": build_image_url_list([p.product_images_1, p.product_images_2, p.product_images_3, p.product_images_4]),
                "heart_cnt": p.heart_cnt,
                "numof_views": p.numof_views,
                "total_sales": p.total_sales,
                "review_cnt": p.review_cnt,
                "review_rating": p.review_rating,
                "reviews": [p.review1, p.review2, p.review3, p.review4, p.review5],
                # "similar_products": [p.similar_product_1, p.similar_product_2, p.similar_product_3],
                "is_like": p.product_code in like_product_codes  # Check if product is liked by the user
            }
            for p in products
        ]
    }

### 5. (POST) 상품 찜하기 ###
@app.post("/api/product_like")
def product_like(user_id: str, product_code: str, db: Session = Depends(get_db)):
    # 이미 찜되었는지 확인하고 추가
    existing_like = db.query(UserFavorite).filter(
        UserFavorite.user_id == user_id,
        UserFavorite.product_code == product_code
    ).first()

    if existing_like:
        # return {"message": "이미 찜되어있는 상품입니다."}
        return {"success": False}
    
    new_favorite = UserFavorite(
        user_id=user_id,
        product_code=product_code
    )
    
    db.add(new_favorite)
    db.commit()
    db.refresh(new_favorite)
    
    # return {"message": "Product liked successfully!", "favorite_id": new_favorite.id}
    return {"success": True}


### 6. (POST) 상품 찜 삭제 ###
@app.post("/api/product_unlike")
def product_unlike(user_id: str, product_code: str, db: Session = Depends(get_db)):
    # 찜 상품 찾기
    favorite_entry = db.query(UserFavorite).filter(
        UserFavorite.user_id == user_id,
        UserFavorite.product_code == product_code
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


### 7. (GET) 장바구니 - 찜 상품 목록 조회 ###
@app.get("/api/cart_list")
def cart_list(user_id: str, db: Session = Depends(get_db)):
    # Join UserFavorite and Product tables to fetch favorite product details
    products = (
        db.query(Product)
        .join(UserFavorite, Product.product_code == UserFavorite.product_code)
        .filter(UserFavorite.user_id == user_id)
        .all()
    )

    return {
        "cart_list": [
            {
                "product_code": p.product_code,
                "detail_url": p.detail_url,
                "product_name": p.product_name,
                # "product_price": p.product_price,
                "discount_rate": p.discount_rate,
                "final_price": p.final_price,
                "brand_name": p.brand_name,
                "brand_image": p.brand_image,
                "category": p.category,
                "category_sub": p.category_sub,
                "product_image": build_image_url(p.product_images_1),
                # "product_images": [p.product_images_1, p.product_images_2, p.product_images_3, p.product_images_4],
                # "heart_cnt": p.heart_cnt,
                # "numof_views": p.numof_views,
                # "total_sales": p.total_sales,
                "review_cnt": p.review_cnt,
                "review_rating": p.review_rating,
                # "reviews": [p.review1, p.review2, p.review3, p.review4, p.review5]
                # "similar_products": [p.similar_product_1, p.similar_product_2, p.similar_product_3]
            }
            for p in products
        ]
    }


### 8. (POST) 장바구니 - 찜 삭제 ###
@app.post("/api/cart_unlike")
def cart_unlike(
    user_id: str, 
    product_codes: List[str] = Body(...),  # Explicitly specify that product_codes comes from the request body
    db: Session = Depends(get_db)
):    
    success_list = []
    failure_list = []
    
    for product_code in product_codes:
        # 찜 상품 찾기
        favorite_entry = db.query(UserFavorite).filter(
            UserFavorite.user_id == user_id,
            UserFavorite.product_code == product_code
        ).first()

        # 못찾으면 실패 리스트에 추가
        if not favorite_entry:
            failure_list.append(product_code)
            continue
        
        # 찾으면 찜 목록에서 삭제
        db.delete(favorite_entry)
        db.commit()
        success_list.append(product_code)
    
    products = (
        db.query(Product)
        .join(UserFavorite, Product.product_code == UserFavorite.product_code)
        .filter(UserFavorite.user_id == user_id)
        .all()
    )
        
    return {"success": True, "removed": success_list, "not_found": failure_list, "cart_list": [
            {
                "product_code": p.product_code,
                "detail_url": p.detail_url,
                "product_name": p.product_name,
                # "product_price": p.product_price,
                "discount_rate": p.discount_rate,
                "final_price": p.final_price,
                "brand_name": p.brand_name,
                "brand_image": p.brand_image,
                "category": p.category,
                "category_sub": p.category_sub,
                "product_image": build_image_url(p.product_images_1),
                # "product_images": [p.product_images_1, p.product_images_2, p.product_images_3, p.product_images_4],
                # "heart_cnt": p.heart_cnt,
                # "numof_views": p.numof_views,
                # "total_sales": p.total_sales,
                "review_cnt": p.review_cnt,
                "review_rating": p.review_rating,
                # "reviews": [p.review1, p.review2, p.review3, p.review4, p.review5]
                # "similar_products": [p.similar_product_1, p.similar_product_2, p.similar_product_3]
            }
            for p in products
        ]
    }


### 9. (GET) 전체 상품 리스트 페이지 - 전체 상품 조회 ###
@app.get("/api/all_product_list") # "/api/highlight_list"
def get_highlight_products(user_id: str, video_id: str, db: Session = Depends(get_db)):
    highlights = db.query(Highlight).filter(Highlight.video_id == video_id).all()
    
    if not highlights:
        return {"message": "No highlights found for this video_id"}
    
    # all main product_code in highlights
    product_codes = [h.product_code for h in highlights]
    # bring all product information using product_code from highlights
    products = {p.product_code: p for p in db.query(Product).filter(Product.product_code.in_(product_codes)).all()}
    
    like_product_codes = {fav.product_code for fav in db.query(UserFavorite).filter(UserFavorite.user_id == user_id).all()}

    return {
        "all_product_list": [
            {
                "highlight_idx": h.highlight_idx,
                "highlight_image_url": h.highlight_image_url,
                "product_code": h.product_code,
                "product_image_url": build_image_url(products[h.product_code].product_images_1),
                "brand_name": products[h.product_code].brand_name,
                "brand_image": build_image_url(products[h.product_code].brand_image),
                "product_name": products[h.product_code].product_name,
                "final_price": products[h.product_code].final_price,
                "discount_rate": products[h.product_code].discount_rate,
                "category": products[h.product_code].category,
                "category_sub": products[h.product_code].category_sub,
                # "find_product_code": [h.product_code, products[h.product_code].similar_product_1, products[h.product_code].similar_product_2, products[h.product_code].similar_product_3]
                # [product_code, similar_1, similar_2, similar_3]
                "is_like": h.product_code in like_product_codes  # Check if product is liked by the user

            }
            for h in highlights
        ]
    }

### 10. INTERNAL STT MODEL EXPERIMENT ### 
# TODO
# Create a Socket.IO asynchronous server with ASGI integration.
sio = socketio.AsyncServer(async_mode="asgi")

# Wrap the FastAPI app with the Socket.IO ASGIApp.
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Initialize the Google Cloud Speech client and set your wake word.
# client = speech.SpeechClient()
wake_word = "새미야"

# Set up credentials for Google Cloud.
# current_directory = os.path.dirname(os.path.abspath(__file__))
# credentials_path = os.path.join(current_directory, "hyub_google_cloud_key.json")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Handle new Socket.IO connections.
@sio.event
async def connect(sid, environ):
    print(f"Socket.IO connected: {sid}")

# Handle Socket.IO disconnections.
@sio.event
async def disconnect(sid):
    print(f"Socket.IO disconnected: {sid}")

# Listen for the 'audio_stream' event from the client.
@sio.event
async def audio_stream(sid, data):
    """
    Expects 'data' to be a bytes-like object containing audio in LINEAR16 format.
    Processes the audio with Google Cloud Speech-to-Text and emits events back.
    """
    # Log the incoming audio stream for debugging purposes.
    print(f"Audio stream received from client {sid} (size: {len(data)} bytes)")

    # try:
    #     # Wrap the incoming audio data for recognition.
    #     audio = speech.RecognitionAudio(content=data)
    #     config = speech.RecognitionConfig(
    #         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #         sample_rate_hertz=16000,
    #         language_code="ko-KR",
    #     )
    #     response = client.recognize(config=config, audio=audio)
    #     sentence = ""
        
    #     # Process the response from Google Cloud.
    #     for result in response.results:
    #         text = result.alternatives[0].transcript
    #         print(f"Recognized: {text}")
    #         sentence += " " + text
    #         if wake_word in text:
    #             # Emit the wake word detected event to the specific client.
    #             await sio.emit("wake_word_detected", room=sid)
        
    #     # Optionally, emit the complete transcription result.
    #     await sio.emit("result_data", {"text": sentence.strip()}, room=sid)
    
    # except Exception as e:
    #     print("Error processing audio:", e)

##################### initialize backend server #####################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
