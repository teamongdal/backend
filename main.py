# backend/main.py using FastAPI

############################################################################################################
############################################################################################################
############################################################################################################
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Form, WebSocket, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

from minsun import minsun_model



# main.py runs on local machine (localhost:8000)
# TODO host online to allow connection from frontend (Azure: app-container? 느낌)

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
def get_videos(user_id: str, db: Session = Depends(get_db)):
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
def get_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.video_id == video_id).first()

    if not video:
        return {"message": "Video not found"}
    
    # return {"video_id": video.video_id, "video_name": video.video_name, "video_url": video.video_url}
    return {"video_url": video.video_url}


### 3. (POST) VOD 재생 페이지 - 유저 음성 발화 상품 검색 + 상품 리스트 조회 ###
# TODO:
# 0. Pass "image" to 민선 model -> gives {bounding_box, feature_vec, color, category, background_removed_image}
# 1. LLM End -> send {capture_image.jpg, llm_keywords_dict} to def co
# 2. 
# UPDATE success -> state: allows us to check where error occurs -> multi-turn
@app.post("/api/search_product")
async def search_product(
    user_id: str,
    audio: UploadFile = File(...),  # Audio file
    image: UploadFile = File(...),  # Image file
    db: Session = Depends(get_db)  # Database session
):
    try:
        # image = r"C:\github\ongdal\backend\local_data\find_product_example_scene.png"
        # audio = r"C:\github\ongdal\backend\tools\sample2.wav"
        # # Ensure at least one file is received
        if not audio and not image:
             return {"success": False, "message": "No audio or image file received"}

        # # Validate audio file format
        if not audio.filename.endswith((".webm", ".mp3", ".wav", ".m4a")): # should be just .wav            
             return {"success": False, "message": "Invalid audio format"}

        # # Validate image file format
        allowed_image_extensions = (".jpg", ".jpeg", ".png")
        if image and not image.filename.endswith(allowed_image_extensions):
            return {"success": False, "message": "Invalid image format"}

        # Read the file into memory (without saving it)
        audio_bytes = await audio.read()
        audio_buffer = io.BytesIO(audio_bytes)  # Convert to a file-like object
        
        # TODO update from .webm to .wav
        if audio.filename.endswith(".webm"):
            wav_buffer = convert_webm_to_wav(audio_buffer)
        else:
            wav_buffer = audio_buffer
        
        ### STT Begin ###
        transcribed_text = process_audio_file(wav_buffer) # transcribed_text ="왼쪽 옷 정보 알려줘"
        ### STT End ###

        ### LLM Begin ###
        llm_keywords = parse_speech_to_json(transcribed_text)
        llm_keywords_dict = json.loads(llm_keywords)
        ### LLM End ###


        ### Detection Model Begin ### - returns {feature_vector, bounding_box, color_vector, output_category}
        # TODO
        # feature_vector, bounding_box, color_vector, output_category = minsun_model(image) # add minsun_model
        feature_vector, bounding_box, color_vector, output_category = minsun_model(image) # add minsun_model
        ## @check if {bounding_box, feature_vector, color_vector, output_category} matches llm_keywords_dict
        ## @if not, return error message -> multi-turn 
        ### Detection Model End ###
        

        ### Recommendation Model Begin ### - returns {recommend_list}
        # TODO
        # finds best product code (comparing feature_vector with all product feature_vectors)
        best_product_code = hyub_sung_model(feature_vector) # add hyub_sung_model
        ### Recommendation Model End ###

        product_entry = db.query(Product).filter(Product.product_code == best_product_code).first()
        display_list = []
        # If a matching entry is found, retrieve similar product codes
        if product_entry:
            display_list = [
                best_product_code,
                product_entry.similar_product_1,
                product_entry.similar_product_2,
                product_entry.similar_product_3
            ]

        products = db.query(Product).filter(Product.product_code.in_(display_list)).all()
        
        # Fetch favorite products for the given user
        like_product_codes = {fav.product_code for fav in db.query(UserFavorite).filter(UserFavorite.user_id == user_id).all()}
        
        print("DONE!")
        print("feature_vector: ", feature_vector)
        print("bounding_box: ", bounding_box)
        print("color_vector: ", color_vector)
        
        return {
            "success": True,
            "message": "Returned Recommendation Succesfully",
            "user_id": user_id,
            "speech_text": transcribed_text,
            "llm_keywords": llm_keywords_dict,
            "product_list": [
                {
                    "product_code": p.product_code,
                    # "detail_url": p.detail_url,
                    "product_name": p.product_name,
                    "product_price": p.product_price,
                    "discount_rate": p.discount_rate,
                    "final_price": p.final_price,
                    "brand_name": p.brand_name,
                    "brand_image": p.brand_image,
                    "category": p.category,
                    "category_sub": p.category_sub,
                    "product_images": [p.product_images_1, p.product_images_2, p.product_images_3, p.product_images_4],
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


# def minsun_model(image: UploadFile):
#     """
#     Receives an image file and LLM keywords dictionary, and returns the feature_vector, bounding_box, color_vector, and output_category
#     """
#     # Read the image file into memory
#     # image_bytes = image.file.read()
#     feature_vector = "feature_vector"
#     return feature_vector


def hyub_sung_model(feature_vector: str):
    """
    Receives feature_vector and returns most similar product_code
    """
    product_code = "cardigan_0022"
    return product_code


### 4. (GET) 상품 검색 결과 페이지 - 상품 리스트 조회 (연동화를 위해 AI 모델 대체하는 함수) ###
## (1) AI 모델에서 product_list(user_id, find_product_code) 함수 호출
## (2) highlight에서는 "/api/product_list (user_id, find_product_code)"로 호출
## TODO remove product_id. use only find_product_id. no query on highlight! 
# def product_list(user_id: str, product_code: Optional[str] = Query(None), find_product_code: Optional[List[str]] = Query(None), db: Session = Depends(get_db)):
@app.get("/api/product_list")
def product_list(user_id: str, product_code: str, db: Session = Depends(get_db)):
    # Fetch products matching find_product_code
    #TODO cancel! if (product_id) -> Query on highlights DB for similar_products_id and set this to find_product_id
    #TODO cancel! if (find_product_id) -> do nothing.
    #TODO cancel! ptimization: check if product_code in highlights DB before running AI model

    # if product_code:
    #     find_product_code = [product_code] + find_product_code
    # else: # dummy find_product_code
    #     find_product_code = ["blazer_0009", "blazer_0012", "blazer_0015", "blazer_0016"] # [product_code + similar_product_code]
    
    if not product_code:
        print("No product code found")
        display_list = ["blazer_0009", "blazer_0012", "blazer_0015", "blazer_0016"] # [product_code + similar_product_code]
    
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
                # "detail_url": p.detail_url,
                "product_name": p.product_name,
                "product_price": p.product_price,
                "discount_rate": p.discount_rate,
                "final_price": p.final_price,
                "brand_name": p.brand_name,
                "brand_image": p.brand_image,
                "category": p.category,
                "category_sub": p.category_sub,
                "product_images": [p.product_images_1, p.product_images_2, p.product_images_3, p.product_images_4],
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

### REMOVE START ###
### 5. (GET) 상품 상세 페이지 - 상품 상세 조회 ### 
# 이건 기존 ("api/product_list")에서 'detail' array에서 이미지 url을 받아와서 보여주기 (frontend 처리)
### REMOVE END ###


### 6. (POST) 상품 찜하기 ###
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


### 7. (POST) 상품 찜 삭제 ###
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


### 8. (GET) 장바구니 - 찜 상품 목록 조회 ###
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
                # "discount_rate": p.discount_rate,
                "final_price": p.final_price,
                "brand_name": p.brand_name,
                "brand_image": p.brand_image,
                "category": p.category,
                "category_sub": p.category_sub,
                # "product_images": [p.product_images_1, p.product_images_2, p.product_images_3, p.product_images_4],
                # "heart_cnt": p.heart_cnt,
                # "numof_views": p.numof_views,
                # "total_sales": p.total_sales,
                # "review_cnt": p.review_cnt,
                # "review_rating": p.review_rating,
                # "reviews": [p.review1, p.review2, p.review3, p.review4, p.review5]
                # "similar_products": [p.similar_product_1, p.similar_product_2, p.similar_product_3]
            }
            for p in products
        ]
    }


### 9. (POST) 장바구니 - 찜 삭제 ###
@app.post("/api/cart_unlike")
def product_unlike(
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
    
    return {"success": True, "removed": success_list, "not_found": failure_list}


### 10. (GET) 전체 상품 리스트 페이지 - 전체 상품 조회 ###
@app.get("/api/all_product_list") # "/api/highlight_list"
def get_highlight_products(video_id: str, db: Session = Depends(get_db)):
    highlights = db.query(Highlight).filter(Highlight.video_id == video_id).all()
    
    if not highlights:
        return {"message": "No highlights found for this video_id"}
    
    # all main product_code in highlights
    product_codes = [h.product_code for h in highlights]
    # bring all product information using product_code from highlights
    products = {p.product_code: p for p in db.query(Product).filter(Product.product_code.in_(product_codes)).all()}
    
    return {
        "all_product_list": [
            {
                "highlight_idx": h.highlight_idx,
                "highlight_image_url": h.highlight_image_url,
                "product_code": h.product_code,
                "product_image_url  ": products[h.product_code].product_images_1,
                "brand_name": products[h.product_code].brand_name,
                "product_name": products[h.product_code].product_name,
                "final_price": products[h.product_code].final_price,
                # "find_product_code": [h.product_code, products[h.product_code].similar_product_1, products[h.product_code].similar_product_2, products[h.product_code].similar_product_3]
                # [product_code, similar_1, similar_2, similar_3]
            }
            for h in highlights
        ]
    }

### 11. (GET) 핸드폰 연동 ### - REMOVE?


### 12. INTERNAL STT MODEL EXPERIMENT ### 
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

##################### AI MODEL #####################

### TODO AI 모델 (1) - LLM (Captured Audio + Video Frame 받아서 LLM 처리) / DB에 있는 제일 유사한 옷의 product_id 반환 ###

### TODO AI 모델 (2) - 제일 유사한 옷의 product_id를 받아 / DB에 있는 유사한 옷 5개의 product_id 반환 (아니면 유사 threshold 넘는 상품들) ###

### TODO AI 모델 (3) - ? ###

##################### initialize backend server #####################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
    # uvicorn.run(host="0.0.0.0", port=8000)