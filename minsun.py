import faiss
import numpy as np
import json
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.models import resnet50
from PIL import Image
import cv2
import os
from roboflow import Roboflow
from sklearn.cluster import KMeans

# Fashion‑CLIP 관련 임포트 (카테고리 분류용)
from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor

# 표준 CLIP 모델을 style attribute 추출용으로 로드 (scripy.py 방식)
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
style_model, style_preprocess = clip.load("ViT-B/32", device=device)

# 텍스트 임베딩 함수 (정규화 포함)
def encode_texts(texts):
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = style_model.encode_text(text_tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# 텍스트 벡터 (각 속성별)
TEXTURE_CATEGORIES = ["rough", "soft", "glossy"]
SEASON_CATEGORIES = ["spring", "summer", "fall", "winter"]
MOOD_CATEGORIES = ["casual", "elegant", "sporty"]

texture_vectors = encode_texts(TEXTURE_CATEGORIES).cpu().numpy()  # shape: (3, 512)
season_vectors  = encode_texts(SEASON_CATEGORIES).cpu().numpy()   # shape: (4, 512)
mood_vectors    = encode_texts(MOOD_CATEGORIES).cpu().numpy()       # shape: (3, 512)

# 이미지의 스타일 임베딩을 추출하는 함수 (scripy.py 방식)
def classify_feature_vector(image, text_vectors):
    image_input = style_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = style_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ torch.tensor(text_vectors, device=device).T)
    return similarity.cpu().numpy().flatten()  # 예: 3, 4, 3 차원 벡터

# --- 설정 ---
# PRODUCT_JSON는 추가된 texture/season/mood vector가 있으므로 수정
FEATURES_NPY = "ver2_2048.npy"  # 2048-dim feature vector
PRODUCT_JSON = "ver2_cat&color&clip.json"  # JSON 파일에 texture_vector, season_vector, mood_vector 포함
ROBOFLOW_API_KEY = "XpCdCNt4CFykx5vyIApu"  # Roboflow API 키
QUERY_IMAGE_PATH = "sky_castle.png"  # 예시 query 이미지 파일 경로
K = 4  # 추천 상위 k개

# 기존의 카테고리 리스트(이제 combined similarity에서 사용하지 않음 – 대신 색상, 텍스처, 시즌, 무드를 사용)
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress",
    "trousers", "skirt", "shorts"
]

# --- 기존 함수들 (faiss_color_cat.py) ---
def load_fashion_clip_model():
    # Fashion‑CLIP 모델을 불러와서 fclip.processor도 저장 (카테고리 분류용)
    fclip = FashionCLIP("fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    fclip.processor = processor
    return fclip

def load_resnet_feature_extractor(device):
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

def load_roboflow_model(api_key):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("deepfashion2-m-11k")
    model = project.version(1).model
    return model

def find_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_feature_for_crop_resnet(cropped_image, feature_extractor, device):
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = feature_extractor(img_tensor)
    return feature.squeeze().cpu().numpy()

# Roboflow 기반 객체 검출 (box와 초기 카테고리 추출)
def detect_query_box(query_image, roboflow_model, score_threshold=0.5):
    temp_path = "temp_query.jpg"
    query_image.save(temp_path)
    predictions = roboflow_model.predict(temp_path, confidence=score_threshold).json()
    os.remove(temp_path)
    if "predictions" not in predictions or len(predictions["predictions"]) == 0:
        width, height = query_image.size
        return [0, 0, width, height], None
    preds = predictions["predictions"]
    preds.sort(key=lambda x: x["confidence"], reverse=True)
    best = preds[0]
    x, y, w, h = best["x"], best["y"], best["width"], best["height"]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return [x1, y1, x2, y2], best.get("class", None)

def detect_query_category_clip(query_image, query_box, fclip, categories):
    cropped = query_image.crop(tuple(map(int, query_box)))
    image_tensor = fclip.processor(images=cropped, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        image_feature = fclip.model.get_image_features(image_tensor)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    text_inputs = fclip.processor(text=categories, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = fclip.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarities = (image_feature @ text_features.T).squeeze(0)
    best_index = similarities.argmax().item()
    return categories[best_index]

def detect_color_kmeans_blur(image, box, k=3):
    x1, y1, x2, y2 = map(int, box)
    image_np = np.array(image)
    cropped_region = image_np[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(cropped_region, (5,5), 0)
    blurred_hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    pixels = blurred_hsv.reshape((-1,3))
    kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return [int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])]

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def map_index_to_json(idx, products):
    count = 0
    for prod_idx, product in enumerate(products):
        clothes = product.get("clothes", [])
        for box_idx, cloth in enumerate(clothes):
            if count == idx:
                return prod_idx, box_idx
            count += 1
    return None, None

def compute_feature(image_path, feature_extractor, device):
    image = find_image(image_path)
    if image is None:
        return None
    feat = extract_feature_for_crop_resnet(image, feature_extractor, device)
    norm = np.linalg.norm(feat)
    if norm == 0:
        norm = 1
    return feat.astype(np.float32) / norm

def minsun_model(image_file, roboflow_api_key="iaKZpe4SwjpsNWkYh7aO") -> tuple:
    # --- 파일 및 설정 ---
    # ver1 단계 (후보 상품 기반 유사도 비교)
    VER1_JSON = "ver2_cat&color&clip.json"
    FEATURES_NPY = "ver2_2048.npy"

    # myeongpum 단계 (후보 상품 선택)
    MYEONGPUM_JSON = "myeongpum_test_cat&color&clip.json"
    MYEONGPUM_FEATURES_NPY = "myeongpum_test_2048.npy"

    # ver1 단계에서 유사 상품 추천 개수
    TOPK_VER1 = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet_model = resnet50(pretrained=True)
    resnet_model.fc = torch.nn.Identity()
    resnet_model.to(device)
    resnet_model.eval()

    # 1. Query 이미지 feature 추출 (ResNet50 기반)
    query_image = find_image(QUERY_IMAGE_PATH)
    if query_image is None:
        exit(1)
    query_feature = extract_feature_for_crop_resnet(query_image, resnet_model, device)
    norm = np.linalg.norm(query_feature)
    if norm == 0:
        norm = 1
    query_feature = query_feature.astype(np.float32) / norm

    # =====================================================
    # [Step 1] myeongpum_test_cat&color&clip.json에서 후보 상품 1개 선택
    # 미리 저장된 myeongpum_test_2048.npy를 사용하여, clothes 필드의 인덱스와 매핑
    # =====================================================
    myeongpum_products = load_json(MYEONGPUM_JSON)

    myeongpum_features = np.load(MYEONGPUM_FEATURES_NPY, allow_pickle=True).astype(np.float32)
    # 정규화
    norms = np.linalg.norm(myeongpum_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    myeongpum_features /= norms

    d = myeongpum_features.shape[1]
    index_myeongpum = faiss.IndexFlatIP(d)
    index_myeongpum.add(myeongpum_features)
    query_feature_reshaped = query_feature.reshape(1, -1)
    distances_myeongpum, indices_myeongpum = index_myeongpum.search(query_feature_reshaped, 1)
    best_idx_myeongpum = indices_myeongpum[0][0]
    prod_idx, box_idx = map_index_to_json(best_idx_myeongpum, myeongpum_products)

    candidate_product = myeongpum_products[prod_idx]
    candidate_similarity = distances_myeongpum[0][0]
    
    # =====================================================
    # [Step 2] ver1_cat&color&clip.json에서 후보 상품의 이미지를 기반으로 상위 3개 유사 상품 선택
    # 미리 저장된 ver1_2048.npy와 clothes 필드 인덱스 매핑 사용
    # =====================================================
    # 후보 상품의 product_images_1에서 feature 추출
    cand_img_path = candidate_product.get("product_images_1")
    candidate_image = find_image(cand_img_path)

    candidate_feature = extract_feature_for_crop_resnet(candidate_image, resnet_model, device)
    norm = np.linalg.norm(candidate_feature)
    if norm == 0:
        norm = 1
    candidate_feature = candidate_feature.astype(np.float32) / norm

    ver1_products = load_json(VER1_JSON)

    ver1_features = np.load(FEATURES_NPY, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(ver1_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    ver1_features /= norms

    d_ver1 = ver1_features.shape[1]
    index_ver1 = faiss.IndexFlatIP(d_ver1)
    index_ver1.add(ver1_features)
    candidate_feature_reshaped = candidate_feature.reshape(1, -1)
    distances_ver1, indices_ver1 = index_ver1.search(candidate_feature_reshaped, TOPK_VER1)
    
    similar_products = []

    for idx, sim in zip(indices_ver1[0], distances_ver1[0]):
        prod_idx, box_idx = map_index_to_json(idx, ver1_products)
        if prod_idx is None:
            continue
        prod = ver1_products[prod_idx]
        similar_products.append(prod)
    
    # =====================================================
    # 최종 결과: myeongpum_test에서 선택된 후보 상품 1개 + ver1에서 선택된 유사 상품 3개 (총 4개)
    # =====================================================
    return candidate_product, similar_products