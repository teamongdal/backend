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
from sklearn.cluster import KMeans
import clip
from difflib import SequenceMatcher

# Fashion‑CLIP 관련 임포트 (카테고리 분류용)
from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor

# 표준 CLIP 모델을 style attribute 추출용으로 로드 (scripy.py 방식)
device = "cuda" if torch.cuda.is_available() else "cpu"
style_model, style_preprocess = clip.load("ViT-B/32", device=device)

# CATEGORIES 수정(협)
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "shorts", "trousers",
    "skirt", "short sleeve dress", "long sleeve dress", "vest dress", 
    "sling dress"]

device = "cuda" if torch.cuda.is_available() else "cpu"
# CLIP 모델 로드 (style attribute 추출용)
style_model, style_preprocess = clip.load("ViT-B/32", device=device)

# 텍스트 임베딩 함수
def encode_texts(texts):
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = style_model.encode_text(text_tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def is_similar_category(cat1, cat2, threshold=0.7):
    """ 두 카테고리의 유사도를 비교하여 일정 임계치 이상이면 동일한 것으로 간주 """
    return SequenceMatcher(None, cat1, cat2).ratio() > threshold

# 카테고리별 텍스처, 시즌, 무드 벡터 (faiss_clip.py 참고)
PATTERN_CATEGORIES = ["Solid color with no Pattern", "Stripe Pattern", "Polka Dot Pattern", "Checkered Pattern"]

pattern_vectors = encode_texts(PATTERN_CATEGORIES).cpu().numpy()    # (4, 512)

def classify_feature_vector(image, text_vectors):
    image_input = style_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = style_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ torch.tensor(text_vectors, device=device).T)
    return similarity.cpu().numpy().flatten()

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def euclidean_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    distance = np.linalg.norm(a - b)  # Euclidean distance 
    # Normalize to similarity score (0 to 1)
    similarity = 1 / (1 + distance)  # The smaller the distance, the closer to 1.
    return similarity

# --- 기본 함수들 ---
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None

def extract_feature_for_crop_resnet(cropped_image, feature_extractor, device):
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img_tensor = transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = feature_extractor(img_tensor)
    return feature.squeeze().cpu().numpy()

def compute_feature(image, feature_extractor, device):
    feat = extract_feature_for_crop_resnet(image, feature_extractor, device)
    norm = np.linalg.norm(feat)
    if norm == 0:
        norm = 1
    return feat.astype(np.float32) / norm

# JSON 내의 clothes box 인덱스 매핑 (faiss_clip, search_dupe 참고)
def map_index_to_json(idx, products):
    count = 0
    for prod_idx, product in enumerate(products):
        clothes = product.get("clothes", [])
        for box_idx, cloth in enumerate(clothes):
            if count == idx:
                return prod_idx, box_idx
            count += 1
    return None, None

def load_detection_model(model_path, device):
    """df2matchrcnn 모델 로드 (DeepFashion2용 Mask R-CNN)"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    from torchvision.models.detection import maskrcnn_resnet50_fpn
    # DeepFashion2의 경우, 배경을 포함해 14개의 클래스로 설정되어 있음
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=14)
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def detect_query_box_df2matchrcnn(query_image, detection_model, direction="middle", score_threshold=0.5):
    # 이미지 전처리
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(query_image).to(device)
    
    with torch.no_grad():
        outputs = detection_model([img_tensor])
    outputs = outputs[0]
    boxes = outputs["boxes"].cpu().numpy()  # shape: (N, 4)
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()  # 클래스 번호
    
    # score threshold 필터링
    valid_indices = np.where(scores > score_threshold)[0]
    if len(valid_indices) == 0:
        width, height = query_image.size
        return [0, 0, width, height], "NoDetection"
    
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]
    
    width, height = query_image.size
    candidates = []
    
    # 방향에 따른 후보 추출
    if direction.lower() == "left":
        # 왼쪽 영역: 이미지의 왼쪽 절반을 대상으로 함
        for i, box in enumerate(boxes):
            center_x = (box[0] + box[2]) / 2
            if center_x < width / 2:
                candidates.append((box, scores[i], labels[i]))
        if candidates:
            # composite score = -x1 + lambda * area
            lambda_val = 0.001  # 면적 가중치 (필요에 따라 조정)
            def composite_score(candidate):
                box, score, label = candidate
                area_val = (box[2] - box[0]) * (box[3] - box[1])
                return -box[0] + lambda_val * area_val
            best_candidate = max(candidates, key=composite_score)
            best_box, best_score, best_label = best_candidate
        else:
            # 후보가 없으면 전체에서 넓이가 가장 큰 것을 선택
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            best_box = boxes[idx]
            best_label = labels[idx]
    elif direction.lower() == "right":
        # 오른쪽 영역: 이미지의 오른쪽 절반을 대상으로 함
        for i, box in enumerate(boxes):
            center_x = (box[0] + box[2]) / 2
            if center_x >= width / 2:
                candidates.append((box, scores[i], labels[i]))
        if candidates:
            # 비슷한 방식으로 오른쪽의 경우는 오른쪽에 가까운(즉, box[2]가 큰) 후보를 선택
            lambda_val = 0.001
            def composite_score_right(candidate):
                box, score, label = candidate
                area_val = (box[2] - box[0]) * (box[3] - box[1])
                # 오른쪽일 경우, 오른쪽 경계에 가까울수록 box[2]가 크므로 그 값을 사용
                return box[2] + lambda_val * area_val
            best_candidate = max(candidates, key=composite_score_right)
            best_box, best_score, best_label = best_candidate
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            best_box = boxes[idx]
            best_label = labels[idx]
    else:  # "middle"인 경우 기존 로직 사용
        for i, box in enumerate(boxes):
            center_x = (box[0] + box[2]) / 2
            if width / 3 <= center_x <= 2 * width / 3:
                candidates.append((box, scores[i], labels[i]))
        if candidates:
            def area(box):
                return (box[2] - box[0]) * (box[3] - box[1])
            best_candidate = max(candidates, key=lambda x: area(x[0]))
            best_box, best_score, best_label = best_candidate
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            best_box = boxes[idx]
            best_label = labels[idx]
    
    best_box = list(map(int, best_box))
    try:
        index = int(best_label) - 1
        if 0 <= index < len(CATEGORIES):
            best_label_str = CATEGORIES[index]
        else:
            best_label_str = str(best_label)
    except Exception as e:
        best_label_str = str(best_label)
    
    # 시각화: query image에 모든 후보들을 초록색, 최고 후보는 파란색으로 표시
    vis_img = cv2.cvtColor(np.array(query_image), cv2.COLOR_RGB2BGR)
    
    return best_box, best_label_str

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

def minsun_model(image_file, direction) -> tuple:
    # --- 설정: 파일 경로 및 변수 ---
    MYEONGPUM_JSON = "myeongpum_test_cat&color&clip&pattern.json"
    MYEONGPUM_FEATURES_NPY = "myeongpum_test_2048.npy"
    VER1_JSON = "ver2_cat&color&clip&pattern.json"
    FEATURES_NPY = "ver2_2048.npy"
    TOPK = 4  # 최종 추천 상위 k개

    image_file = "our_year_2.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet_model = resnet50(pretrained=True)
    resnet_model.fc = torch.nn.Identity()
    resnet_model.to(device)
    resnet_model.eval()

     # --------------------------
    # 1. Query 이미지 로드 및 feature 추출
    # --------------------------
    query_img = find_image(image_file)
    query_feature = compute_feature(query_img, resnet_model, device)
    
    # --------------------------
    # 2. 객체 탐지
    # --------------------------
    df2_model = load_detection_model("df2matchrcnn", device)  # df2matchrcnn checkpoint 경로 필요

    # 모델로 객체 탐지 (방향 "middle" 예시)
    box_df2, label_df2 = detect_query_box_df2matchrcnn(query_img, df2_model, direction=direction, score_threshold=0.5)

    crop_img = query_img.crop(box_df2)
    _, final_label = detect_query_box_df2matchrcnn(crop_img, df2_model, direction=direction, score_threshold=0.5)
    final_color = detect_color_kmeans_blur(query_img, box_df2)

    # --------------------------
    # 3. 후보 상품 선택 (myeongpum_test 단계) - Stable Diffusion VAE latent embedding 방식 적용
    # --------------------------
    # (이미지 전처리 및 norm 계산은 기존과 동일)
    myeongpum_products = load_json(MYEONGPUM_JSON)
    if not os.path.exists(MYEONGPUM_FEATURES_NPY):
        print("myeongpum_test_2048.npy 파일이 존재하지 않습니다.")
        exit(1)
    myeongpum_features = np.load(MYEONGPUM_FEATURES_NPY, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(myeongpum_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    myeongpum_features /= norms

    disallowed = {"skirt", "shorts", "trousers"}
    query_category = final_label.lower()  # 객체 탐지된 query 영역의 카테고리

    # allowed mapping: 각 상품의 clothes 중, category가 query_category와 일치하는 항목만 선택
    allowed_global_indices = []
    allowed_mapping = []  # (prod_idx, box_idx)
    global_idx = 0
    for prod_idx, product in enumerate(myeongpum_products):
        for box_idx, cloth in enumerate(product.get("clothes", [])):
            cat = cloth.get("category", "").lower()
            if query_category in {"long sleeve top", "long sleeve outwear"}:
                if cat in {"long sleeve top", "long sleeve outwear"}:
                    allowed_global_indices.append(global_idx)
                    allowed_mapping.append((prod_idx, box_idx))
            elif query_category in {"short sleeve top", "short sleeve outwear"}:
                if cat in {"short sleeve top", "short sleeve outwear"}:
                    allowed_global_indices.append(global_idx)
                    allowed_mapping.append((prod_idx, box_idx))
            else:
                if cat == query_category:
                    allowed_global_indices.append(global_idx)
                    allowed_mapping.append((prod_idx, box_idx))
            global_idx += 1

    # 후보 상품 선택: combined similarity 방식으로 점수를 계산
    # 하이퍼파라미터 설정 (추천 상품 검색과 동일)
    alpha = 0.5     # feature similarity
    beta = 0.1      # color similarity
    zeta = 0.3      # category similarity
    theta = 0.5     # pattern similarity

    # query_feature는 앞서 추출한 값 사용.
    # query 색상은 detect_color_kmeans_blur()에서 얻은 final_color 사용.
    # query의 texture, season, mood는 객체 탐지된 영역 crop_img에서 추출.
    query_pattern = classify_feature_vector(crop_img, pattern_vectors)

    candidate_scores = []
    candidate_details = []  # (global_idx, prod_idx, box_idx)
    # query_category는 최종 감지된 category (소문자)
    for i in range(len(allowed_global_indices)):
        global_idx = allowed_global_indices[i]
        prod_idx, box_idx = allowed_mapping[i]
        candidate_feat = myeongpum_features[global_idx]
        feat_sim = cosine_similarity(query_feature, candidate_feat)
        product = myeongpum_products[prod_idx]
        cloth = product.get("clothes", [])[box_idx]
        # 색상 유사도 계산
        candidate_color = cloth.get("color_vector")
        if candidate_color is None:
            color_sim = 0
        else:
            candidate_color = np.array(candidate_color, dtype=float)
            norm_candidate = np.linalg.norm(candidate_color)
            if norm_candidate == 0:
                norm_candidate = 1
            candidate_color = candidate_color / norm_candidate
            color_sim = cosine_similarity(final_color, candidate_color)

        # category similarity (clothes 내 category 비교)
        candidate_category = cloth.get("category", "").lower()
        category_sim = 1 if candidate_category == query_category else 0

        # pattern similarity 계산 (new)
        prod_pattern = cloth.get("pattern_vector")
        if (query_pattern is not None) and (prod_pattern is not None):
            query_pattern_arr = np.array(query_pattern, dtype=float)
            prod_pattern_arr = np.array(prod_pattern, dtype=float)
            # Check if the highest value indices match; if not, skip this candidate.
            if np.argmax(query_pattern_arr) != np.argmax(prod_pattern_arr):
                continue  # Skip candidate product due to pattern mismatch.
            else:
                pattern_sim = cosine_similarity(query_pattern_arr, prod_pattern_arr)
        else:
            pattern_sim = 0

        combined = (alpha * feat_sim +
                    beta * color_sim +
                    zeta * category_sim +
                    theta * pattern_sim)
        candidate_scores.append(combined)
        candidate_details.append((global_idx, prod_idx, box_idx))
    
    # 후보 중 결합 유사도가 가장 높은 것을 선택
    best_idx = np.argmax(candidate_scores)
    selected_global_idx, selected_prod_idx, selected_box_idx = candidate_details[best_idx]
    candidate_product = myeongpum_products[selected_prod_idx]
    candidate_sim = candidate_scores[best_idx]
    candidate_img = find_image(candidate_product.get("product_images_1"))
    candidate_box_idx = selected_box_idx  # 이제 candidate_box_idx가 정의됨
    candidate_product_code = candidate_product.get("product_code")

    # --------------------------
    # 4. 추천 상품 검색 (ver2 단계) – candidate_product의 category_sub 기준으로 allowed 항목 필터링
    ver1_products = load_json(VER1_JSON)
    if not os.path.exists(FEATURES_NPY):
        print("ver2_2048.npy 파일이 존재하지 않습니다.")
        exit(1)
    ver1_features = np.load(FEATURES_NPY, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(ver1_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    ver1_features /= norms

    # candidate_product의 category_sub
    candidate_cat_sub = candidate_product.get("category_sub", "").lower()
    query_pattern = classify_feature_vector(candidate_img, pattern_vectors)

    # allowed mapping for ver1_products based on candidate_product's category_sub
    allowed_global_indices_v1 = []
    allowed_mapping_v1 = []  # 각 인덱스에 대해 (prod_idx, box_idx)
    global_idx = 0
    for prod_idx, product in enumerate(ver1_products):
        product_cat_sub = product.get("category_sub", "")
        # 제품 레벨의 category_sub가 candidate와 일치하지 않으면 건너뜁니다.
        if product_cat_sub == candidate_cat_sub or is_similar_category(product_cat_sub, candidate_cat_sub):
            for box_idx, cloth in enumerate(product.get("clothes", [])):
                allowed_global_indices_v1.append(global_idx)
                allowed_mapping_v1.append((prod_idx, box_idx))
                global_idx += 1
        else:
            global_idx += len(product.get("clothes", []))  # 일치하지 않으면 건너뜀

    # allowed_features_v1는 allowed_global_indices_v1에 해당하는 feature만 포함
    allowed_features_v1 = ver1_features[allowed_global_indices_v1]
    index_v1_allowed = faiss.IndexFlatIP(allowed_features_v1.shape[1])
    index_v1_allowed.add(allowed_features_v1)
    TOPK = 4

    # 후보 상품 이미지 feature는 이미 candidate_feat로 추출됨

    try:
        color_vec = candidate_product["clothes"][candidate_box_idx]["color_vector"]
        query_color_rep = np.array(color_vec, dtype=float)
        norm_color = np.linalg.norm(query_color_rep)
        if norm_color == 0:
            norm_color = 1
        query_color_rep = query_color_rep / norm_color
    except Exception as e:
        query_color_rep = np.zeros(3)

    query_pattern = classify_feature_vector(candidate_img, pattern_vectors)

    # 기존 하이퍼파라미터
    alpha = 0.5     # feature similarity
    beta = 0.4      # color similarity
    zeta = 0.3      # category similarity
    theta = 0.5     # pattern similarity

    allowed_combined_scores = []
    candidate_category = candidate_product["clothes"][candidate_box_idx].get("category")
    for idx in range(len(allowed_global_indices_v1)):
        global_idx = allowed_global_indices_v1[idx]
        prod_idx, box_idx = allowed_mapping_v1[idx]
        feat_sim = cosine_similarity(candidate_feat.flatten(), ver1_features[global_idx])
        product = ver1_products[prod_idx]
        prod_cloth = product.get("clothes", [])[box_idx]
        
        # color similarity 계산
        prod_color = prod_cloth.get("color_vector")
        if prod_color is None:
            color_sim = 0
        else:
            prod_color = np.array(prod_color, dtype=float)
            norm_prod = np.linalg.norm(prod_color)
            if norm_prod == 0:
                norm_prod = 1
            prod_color = prod_color / norm_prod
            color_sim = cosine_similarity(query_color_rep, prod_color)
        
        # category similarity 계산
        prod_cat = prod_cloth.get("category")
        category_sim = 1 if prod_cat == candidate_category else 0
        
        # pattern similarity 계산 (new)
        prod_pattern = prod_cloth.get("pattern_vector")
        if (query_pattern is not None) and (prod_pattern is not None):
            query_pattern_arr = np.array(query_pattern, dtype=float)
            prod_pattern_arr = np.array(prod_pattern, dtype=float)
            # Check if the highest value indices match; if not, skip this candidate.
            if np.argmax(query_pattern_arr) != np.argmax(prod_pattern_arr):
                continue  # Skip candidate product due to pattern mismatch.
            else:
                pattern_sim = cosine_similarity(query_pattern_arr, prod_pattern_arr)
        else:
            pattern_sim = 0
        
        combined = (alpha * feat_sim +
                    beta * color_sim +
                    zeta * category_sim +
                    theta * pattern_sim)
        allowed_combined_scores.append(combined)

    sorted_indices = np.argsort(allowed_combined_scores)[::-1]
    mapping = []
    seen_codes = set()
    for idx in sorted_indices:
        prod_idx, box_idx = allowed_mapping_v1[idx]
        prod_code = ver1_products[prod_idx].get("product_code", "NoName")
        if prod_code in seen_codes:
            continue
        seen_codes.add(prod_code)
        # mapping에는 allowed_combined_scores의 인덱스(idx)를 저장합니다.
        mapping.append((idx, prod_idx, box_idx))
        if len(mapping) >= TOPK:
            break
    
    dupe_code = []

    # 추천 상품 시각화 (allowed 항목 기준)
    for score_idx, p_idx, b_idx in mapping:
        product = ver1_products[p_idx]
        img_path = product.get("product_images_1")
        if not img_path:
            continue
        if not img_path.lower().endswith(".jpg"):
            img_path += ".jpg"
        prod_img = cv2.imread(img_path)
        if prod_img is None:
            continue
        # allowed clothes 항목 찾기
        allowed_cloth = None
        for cloth in product.get("clothes", []):
            if cloth.get("category", "").lower() not in disallowed:
                allowed_cloth = cloth
                break
        if allowed_cloth is None:
            continue
        box = allowed_cloth.get("box")
        score = allowed_cloth.get("score", 0)
        prod_cat = allowed_cloth.get("category", "unknown")
        prod_code = product.get("product_code", "NoName")
        dupe_code.append(prod_code)

    return [candidate_product_code], dupe_code
    # return ["manual_blazer_0001"], ["manual_blazer_0005", "manual_blazer_0003", "manual_blazer_0004","manual_blazer_0010"]

# if __name__ == "__main__":
#     result, dupe = minsun_model("tools/find_product_example_scene.png", "left")
#     print(result)
#     print(dupe)